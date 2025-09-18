"""
Confidence-guided adaptive sampling for Graph-DiT
Dynamically adjusts sampling steps based on model confidence
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple
import utils
from diffusion import diffusion_utils
from diffusion.fast_sampler import DDIMSampler


class ConfidenceAdaptiveSampler(DDIMSampler):
    """
    Adaptive sampler that adjusts timesteps based on model prediction confidence.
    High confidence → skip steps for speed
    Low confidence → use more steps for quality
    """

    def __init__(self, model, noise_schedule, transition_model,
                 fast_steps: int = 50, eta: float = 0.0,
                 min_steps: int = 20, max_steps: int = 100,
                 confidence_threshold_high: float = 0.85,
                 confidence_threshold_low: float = 0.6,
                 skip_factor: float = 2.0):
        """
        Args:
            min_steps: Minimum number of sampling steps
            max_steps: Maximum number of sampling steps
            confidence_threshold_high: Skip steps above this confidence
            confidence_threshold_low: Add steps below this confidence
            skip_factor: Factor for step skipping/adding
        """
        super().__init__(model, noise_schedule, transition_model, fast_steps, eta)

        self.min_steps = min_steps
        self.max_steps = max_steps
        self.confidence_threshold_high = confidence_threshold_high
        self.confidence_threshold_low = confidence_threshold_low
        self.skip_factor = skip_factor

    def _compute_prediction_confidence(self, pred_X: torch.Tensor, pred_E: torch.Tensor,
                                     node_mask: torch.Tensor) -> float:
        """
        Compute model confidence based on prediction entropy

        Lower entropy → Higher confidence
        Higher entropy → Lower confidence
        """
        # Compute entropy for node predictions
        entropy_X = -torch.sum(pred_X * torch.log(pred_X + 1e-8), dim=-1)
        entropy_X = entropy_X[node_mask].mean().item()

        # Compute entropy for edge predictions
        edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        entropy_E = -torch.sum(pred_E * torch.log(pred_E + 1e-8), dim=-1)
        entropy_E = entropy_E[edge_mask].mean().item()

        # Combine entropies and convert to confidence
        total_entropy = (entropy_X + entropy_E) / 2
        max_entropy = np.log(max(pred_X.shape[-1], pred_E.shape[-1]))
        confidence = 1.0 - (total_entropy / max_entropy)

        return max(0.0, min(1.0, confidence))

    def _compute_step_adjustment(self, confidence: float, current_step: int,
                               total_steps: int) -> int:
        """
        Compute step adjustment based on confidence

        Returns:
            step_adjustment: positive = skip steps, negative = add steps
        """
        if confidence > self.confidence_threshold_high:
            # High confidence: skip steps (but not too many)
            max_skip = min(3, (total_steps - current_step) // 4)
            skip_amount = int(self.skip_factor * (confidence - self.confidence_threshold_high) * 10)
            return min(skip_amount, max_skip)

        elif confidence < self.confidence_threshold_low:
            # Low confidence: add steps (interpolate additional timesteps)
            add_amount = int(self.skip_factor * (self.confidence_threshold_low - confidence) * 5)
            return -min(add_amount, 2)  # Don't add too many steps

        else:
            # Medium confidence: continue normally
            return 0

    def sample_adaptive(self, batch_size: int, y: torch.Tensor, node_mask: torch.Tensor,
                       guidance_scale: Optional[float] = None,
                       device: str = 'cuda') -> List:
        """
        Adaptive sampling with confidence-based step adjustment
        """
        # Initialize from noise
        z_T = diffusion_utils.sample_discrete_feature_noise(
            limit_dist=self.model.limit_dist, node_mask=node_mask
        )
        X, E = z_T.X.to(device), z_T.E.to(device)

        # Start with base timesteps
        timesteps_list = self.timesteps.tolist()

        # Adaptive sampling loop
        i = 0
        confidence_history = []
        step_adjustments = []

        while i < len(timesteps_list) and len(confidence_history) < self.max_steps:
            t = timesteps_list[i]
            t_batch = torch.tensor([t]).repeat(batch_size, 1).float().to(device)

            # Get next timestep
            if i < len(timesteps_list) - 1:
                t_prev = timesteps_list[i + 1]
            else:
                t_prev = 0

            t_prev_batch = torch.tensor([t_prev]).repeat(batch_size, 1).float().to(device)

            # Predict and compute confidence
            noisy_data = {'X_t': X, 'E_t': E, 'y_t': y, 't': t_batch / self.noise_schedule.timesteps, 'node_mask': node_mask}
            pred = self.model.forward(noisy_data, unconditioned=False)
            pred_X = F.softmax(pred.X, dim=-1)
            pred_E = F.softmax(pred.E, dim=-1)

            # Compute confidence
            confidence = self._compute_prediction_confidence(pred_X, pred_E, node_mask)
            confidence_history.append(confidence)

            # Perform denoising step
            X, E = self._ddim_step(
                X, E, y, t_batch / self.noise_schedule.timesteps,
                t_prev_batch / self.noise_schedule.timesteps,
                node_mask, guidance_scale
            )

            # Adaptive step adjustment
            if i < len(timesteps_list) - 3:  # Don't adjust near the end
                step_adj = self._compute_step_adjustment(
                    confidence, len(confidence_history), self.max_steps
                )
                step_adjustments.append(step_adj)

                if step_adj > 0:
                    # Skip steps (high confidence)
                    i += min(step_adj + 1, len(timesteps_list) - i)
                elif step_adj < 0:
                    # Add intermediate steps (low confidence)
                    if i < len(timesteps_list) - 1:
                        current_t = timesteps_list[i]
                        next_t = timesteps_list[i + 1]
                        # Add intermediate timestep
                        intermediate_t = (current_t + next_t) // 2
                        timesteps_list.insert(i + 1, intermediate_t)
                    i += 1
                else:
                    # Normal progression
                    i += 1
            else:
                i += 1

            # Safety check to prevent infinite loops
            if len(confidence_history) >= self.max_steps:
                break

        # Convert to molecule format
        molecules = self._to_molecules(X, E, node_mask)

        # Return additional info about the adaptive process
        return {
            'molecules': molecules,
            'confidence_history': confidence_history,
            'step_adjustments': step_adjustments,
            'total_steps': len(confidence_history),
            'avg_confidence': np.mean(confidence_history),
        }

    def sample_fast(self, batch_size: int, y: torch.Tensor, node_mask: torch.Tensor,
                   guidance_scale: Optional[float] = None,
                   device: str = 'cuda') -> List:
        """Wrapper to maintain compatibility with base class"""
        result = self.sample_adaptive(batch_size, y, node_mask, guidance_scale, device)
        return result['molecules']


class UncertaintyEstimator:
    """
    Estimates model uncertainty using ensemble or dropout-based methods
    """

    def __init__(self, model, num_samples: int = 5):
        self.model = model
        self.num_samples = num_samples

    def estimate_uncertainty(self, noisy_data: dict) -> Tuple[torch.Tensor, float]:
        """
        Estimate prediction uncertainty using Monte Carlo Dropout

        Returns:
            mean_prediction: Average prediction
            uncertainty_score: Uncertainty score (0-1)
        """
        predictions = []

        # Enable dropout for uncertainty estimation
        self.model.train()

        with torch.no_grad():
            for _ in range(self.num_samples):
                pred = self.model.forward(noisy_data, unconditioned=False)
                pred_X = F.softmax(pred.X, dim=-1)
                pred_E = F.softmax(pred.E, dim=-1)
                predictions.append((pred_X, pred_E))

        # Compute mean and variance
        pred_X_stack = torch.stack([p[0] for p in predictions])
        pred_E_stack = torch.stack([p[1] for p in predictions])

        mean_X = pred_X_stack.mean(dim=0)
        mean_E = pred_E_stack.mean(dim=0)

        var_X = pred_X_stack.var(dim=0).mean().item()
        var_E = pred_E_stack.var(dim=0).mean().item()

        uncertainty_score = (var_X + var_E) / 2

        self.model.eval()

        return (mean_X, mean_E), uncertainty_score


class MultiScaleConfidenceSampler(ConfidenceAdaptiveSampler):
    """
    Advanced sampler that uses multiple confidence measures
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uncertainty_estimator = UncertaintyEstimator(self.model)

    def _compute_multiscale_confidence(self, pred_X: torch.Tensor, pred_E: torch.Tensor,
                                     node_mask: torch.Tensor, noisy_data: dict) -> float:
        """
        Compute confidence using multiple measures:
        1. Prediction entropy (existing)
        2. Model uncertainty via dropout
        3. Consistency across time
        """
        # Base entropy confidence
        entropy_confidence = self._compute_prediction_confidence(pred_X, pred_E, node_mask)

        # Uncertainty-based confidence
        _, uncertainty = self.uncertainty_estimator.estimate_uncertainty(noisy_data)
        uncertainty_confidence = 1.0 - min(1.0, uncertainty * 10)  # Scale uncertainty

        # Weighted combination
        final_confidence = 0.7 * entropy_confidence + 0.3 * uncertainty_confidence

        return max(0.0, min(1.0, final_confidence))

    def sample_adaptive(self, *args, **kwargs):
        """Override to use multiscale confidence"""
        # Replace confidence computation method temporarily
        original_method = self._compute_prediction_confidence
        self._compute_prediction_confidence = self._compute_multiscale_confidence

        try:
            result = super().sample_adaptive(*args, **kwargs)
        finally:
            # Restore original method
            self._compute_prediction_confidence = original_method

        return result