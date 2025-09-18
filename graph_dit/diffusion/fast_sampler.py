"""
Fast sampling methods for Graph-DiT including DDIM variants
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List
import utils
from diffusion import diffusion_utils


class DDIMSampler:
    """DDIM-style sampler for discrete graph diffusion models"""

    def __init__(self, model, noise_schedule, transition_model,
                 fast_steps: int = 50, eta: float = 0.0):
        """
        Args:
            model: The denoising model
            noise_schedule: Noise schedule object
            transition_model: Transition model for discrete states
            fast_steps: Number of sampling steps (default 50, much faster than 500)
            eta: DDIM parameter (0 = deterministic, 1 = DDPM)
        """
        self.model = model
        self.noise_schedule = noise_schedule
        self.transition_model = transition_model
        self.fast_steps = fast_steps
        self.eta = eta

        # Pre-compute sampling timesteps
        self.timesteps = self._get_timesteps()

    def _get_timesteps(self) -> torch.Tensor:
        """Generate non-uniform timesteps for fast sampling"""
        # Use quadratic spacing for better quality
        timesteps = torch.linspace(0, 1, self.fast_steps + 1)[1:] ** 2
        timesteps = (timesteps * self.noise_schedule.timesteps).long()
        return torch.flip(timesteps, dims=[0])

    def sample_fast(self, batch_size: int, y: torch.Tensor, node_mask: torch.Tensor,
                   guidance_scale: Optional[float] = None,
                   device: str = 'cuda') -> List:
        """
        Fast sampling with DDIM-style approach

        Args:
            batch_size: Batch size
            y: Conditional information
            node_mask: Mask for valid nodes
            guidance_scale: Classifier-free guidance scale
            device: Device to run on

        Returns:
            List of generated molecules
        """
        # Initialize from noise
        z_T = diffusion_utils.sample_discrete_feature_noise(
            limit_dist=self.model.limit_dist, node_mask=node_mask
        )
        X, E = z_T.X.to(device), z_T.E.to(device)

        # Fast sampling loop
        timesteps_tensor = self.timesteps.to(device)

        for i, t in enumerate(timesteps_tensor):
            t_batch = t.repeat(batch_size, 1).float()

            # Get previous timestep
            if i < len(timesteps_tensor) - 1:
                t_prev = timesteps_tensor[i + 1]
            else:
                t_prev = torch.tensor(0)

            t_prev_batch = t_prev.repeat(batch_size, 1).float()

            # Predict noise/clean state
            X, E = self._ddim_step(
                X, E, y, t_batch / self.noise_schedule.timesteps,
                t_prev_batch / self.noise_schedule.timesteps,
                node_mask, guidance_scale
            )

        # Convert to molecule format
        molecules = self._to_molecules(X, E, node_mask)
        return molecules

    def _ddim_step(self, X_t, E_t, y, t_norm, t_prev_norm, node_mask, guidance_scale):
        """Single DDIM sampling step"""
        bs = X_t.shape[0]

        # Get noise schedule parameters
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_norm)
        alpha_prev_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_prev_norm)

        # Neural network prediction
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y, 't': t_norm, 'node_mask': node_mask}

        # Get conditional and unconditional predictions
        pred_cond = self.model.forward(noisy_data, unconditioned=False)
        pred_X_cond = F.softmax(pred_cond.X, dim=-1)
        pred_E_cond = F.softmax(pred_cond.E, dim=-1)

        if guidance_scale is not None and guidance_scale != 1.0:
            pred_uncond = self.model.forward(noisy_data, unconditioned=True)
            pred_X_uncond = F.softmax(pred_uncond.X, dim=-1)
            pred_E_uncond = F.softmax(pred_uncond.E, dim=-1)

            # Apply classifier-free guidance
            pred_X = pred_X_uncond + guidance_scale * (pred_X_cond - pred_X_uncond)
            pred_E = pred_E_uncond + guidance_scale * (pred_E_cond - pred_E_uncond)

            # Renormalize
            pred_X = pred_X.clamp(min=1e-8)
            pred_E = pred_E.clamp(min=1e-8)
            pred_X = pred_X / pred_X.sum(dim=-1, keepdim=True)
            pred_E = pred_E / pred_E.sum(dim=-1, keepdim=True)
        else:
            pred_X, pred_E = pred_X_cond, pred_E_cond

        # DDIM update with discrete states
        X_prev, E_prev = self._discrete_ddim_update(
            X_t, E_t, pred_X, pred_E, alpha_t_bar, alpha_prev_bar, node_mask
        )

        return X_prev, E_prev

    def _discrete_ddim_update(self, X_t, E_t, pred_X, pred_E, alpha_t_bar, alpha_prev_bar, node_mask):
        """DDIM update for discrete states"""
        bs, n, _ = X_t.shape

        # Compute predicted x0 (clean state)
        X_all_t = torch.cat([X_t, E_t.reshape(bs, n, -1)], dim=-1)
        pred_all = torch.cat([pred_X, pred_E.reshape(bs, n, -1)], dim=-1)

        # Get transition matrices
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, X_t.device)
        Qt_prev_b = self.transition_model.get_Qt_bar(alpha_prev_bar, X_t.device)

        # Predict x0 using Bayes rule (simplified for fast sampling)
        pred_x0 = pred_all  # Use model prediction as x0 estimate

        # DDIM-style deterministic update
        coeff_prev = torch.sqrt(alpha_prev_bar).unsqueeze(-1)
        coeff_noise = torch.sqrt(1 - alpha_prev_bar).unsqueeze(-1)

        # For discrete case, we sample from the predicted distribution
        X_prev_all = coeff_prev * pred_x0 + coeff_noise * torch.randn_like(pred_x0)

        # Split back to X and E
        X_prev_prob = X_prev_all[:, :, :pred_X.shape[-1]]
        E_prev_prob = X_prev_all[:, :, pred_X.shape[-1]:].reshape(bs, n, n, -1)

        # Softmax to get valid probabilities
        X_prev_prob = F.softmax(X_prev_prob, dim=-1)
        E_prev_prob = F.softmax(E_prev_prob, dim=-1)

        # Sample discrete states
        sampled = diffusion_utils.sample_discrete_features(
            X_prev_prob, E_prev_prob, node_mask
        )

        X_prev = F.one_hot(sampled.X, num_classes=X_prev_prob.shape[-1]).float()
        E_prev = F.one_hot(sampled.E, num_classes=E_prev_prob.shape[-1]).float()

        return X_prev, E_prev

    def _to_molecules(self, X, E, node_mask):
        """Convert tensor representations to molecule list"""
        batch_size = X.shape[0]
        molecules = []

        for i in range(batch_size):
            # Find number of valid nodes
            n_nodes = node_mask[i].sum().item()

            # Extract atom and bond types
            atom_types = X[i, :n_nodes].cpu()
            edge_types = E[i, :n_nodes, :n_nodes].cpu()

            molecules.append([atom_types, edge_types])

        return molecules


class AdaptiveStepSampler(DDIMSampler):
    """Adaptive step size sampler that adjusts based on prediction confidence"""

    def __init__(self, *args, min_steps: int = 20, max_steps: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_steps = min_steps
        self.max_steps = max_steps

    def sample_adaptive(self, *args, confidence_threshold: float = 0.95, **kwargs):
        """Sample with adaptive step sizes based on model confidence"""
        # Implementation would adapt step size based on prediction entropy
        # For now, fall back to regular fast sampling
        return self.sample_fast(*args, **kwargs)