"""
Contrastive Molecular Representation Learning for Graph-DiT
Self-supervised learning through molecular augmentation and contrastive loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
import utils


class MolecularAugmenter:
    """
    Applies various augmentations to molecular graphs for contrastive learning
    """

    def __init__(self, augmentation_config: Dict = None):
        """
        Args:
            augmentation_config: Configuration for different augmentation strategies
        """
        if augmentation_config is None:
            augmentation_config = {
                'node_dropout': 0.1,
                'edge_perturbation': 0.1,
                'node_feature_noise': 0.05,
                'subgraph_removal': 0.05,
                'atom_masking': 0.1
            }

        self.config = augmentation_config

    def augment_molecule(self, atom_types: torch.Tensor, edge_types: torch.Tensor,
                        node_mask: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply random augmentations to a molecular graph

        Args:
            atom_types: [N, atom_dim] atom type features
            edge_types: [N, N, edge_dim] edge type features
            node_mask: [N] node mask
            y: [y_dim] molecular properties

        Returns:
            Augmented molecule components
        """
        # Clone to avoid modifying original
        aug_atom_types = atom_types.clone()
        aug_edge_types = edge_types.clone()
        aug_node_mask = node_mask.clone()
        aug_y = y.clone()

        # Randomly select augmentation strategies
        augmentations = []
        if random.random() < 0.3:
            augmentations.append('node_dropout')
        if random.random() < 0.3:
            augmentations.append('edge_perturbation')
        if random.random() < 0.4:
            augmentations.append('node_feature_noise')
        if random.random() < 0.2:
            augmentations.append('atom_masking')

        # Apply selected augmentations
        for aug_type in augmentations:
            if aug_type == 'node_dropout':
                aug_atom_types, aug_edge_types, aug_node_mask = self._node_dropout(
                    aug_atom_types, aug_edge_types, aug_node_mask
                )
            elif aug_type == 'edge_perturbation':
                aug_edge_types = self._edge_perturbation(aug_edge_types, aug_node_mask)
            elif aug_type == 'node_feature_noise':
                aug_atom_types = self._node_feature_noise(aug_atom_types, aug_node_mask)
            elif aug_type == 'atom_masking':
                aug_atom_types = self._atom_masking(aug_atom_types, aug_node_mask)

        return aug_atom_types, aug_edge_types, aug_node_mask, aug_y

    def _node_dropout(self, atom_types: torch.Tensor, edge_types: torch.Tensor,
                     node_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly drop nodes (and their connections)"""
        n_nodes = node_mask.sum().item()
        if n_nodes <= 2:  # Don't drop if too few nodes
            return atom_types, edge_types, node_mask

        # Randomly select nodes to drop
        n_drop = max(1, int(n_nodes * self.config['node_dropout']))
        valid_indices = torch.where(node_mask)[0]
        drop_indices = valid_indices[torch.randperm(len(valid_indices))[:n_drop]]

        # Update node mask
        new_node_mask = node_mask.clone()
        new_node_mask[drop_indices] = False

        # Update edge matrix (remove connections to/from dropped nodes)
        new_edge_types = edge_types.clone()
        for idx in drop_indices:
            new_edge_types[idx, :] = 0
            new_edge_types[:, idx] = 0
            # Set to 'no bond' type
            new_edge_types[idx, :, 0] = 1
            new_edge_types[:, idx, 0] = 1

        return atom_types, new_edge_types, new_node_mask

    def _edge_perturbation(self, edge_types: torch.Tensor,
                          node_mask: torch.Tensor) -> torch.Tensor:
        """Randomly perturb edge types"""
        new_edge_types = edge_types.clone()
        n_nodes = node_mask.sum().item()

        # Find existing edges
        existing_edges = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if node_mask[i] and node_mask[j]:
                    # Check if there's a bond (not 'no bond' type)
                    if edge_types[i, j, 0] < 0.9:  # Not 'no bond'
                        existing_edges.append((i, j))

        if not existing_edges:
            return edge_types

        # Randomly perturb some edges
        n_perturb = max(1, int(len(existing_edges) * self.config['edge_perturbation']))
        perturb_edges = random.sample(existing_edges, min(n_perturb, len(existing_edges)))

        for i, j in perturb_edges:
            # Get current bond type
            current_bond = torch.argmax(edge_types[i, j]).item()

            # Randomly change to a different bond type (but not 'no bond')
            available_bonds = [1, 2, 3, 4]  # Single, double, triple, aromatic
            if current_bond in available_bonds:
                available_bonds.remove(current_bond)

            if available_bonds:
                new_bond = random.choice(available_bonds)

                # Update edge matrix (symmetric)
                new_edge_types[i, j] = 0
                new_edge_types[j, i] = 0
                new_edge_types[i, j, new_bond] = 1
                new_edge_types[j, i, new_bond] = 1

        return new_edge_types

    def _node_feature_noise(self, atom_types: torch.Tensor,
                           node_mask: torch.Tensor) -> torch.Tensor:
        """Add noise to node features"""
        new_atom_types = atom_types.clone()
        noise_level = self.config['node_feature_noise']

        # Add Gaussian noise to atom features
        noise = torch.randn_like(new_atom_types) * noise_level
        new_atom_types = new_atom_types + noise

        # Apply only to valid nodes
        new_atom_types = new_atom_types * node_mask.unsqueeze(-1)

        # Renormalize to maintain probability distribution
        new_atom_types = F.softmax(new_atom_types, dim=-1)

        return new_atom_types

    def _atom_masking(self, atom_types: torch.Tensor,
                     node_mask: torch.Tensor) -> torch.Tensor:
        """Randomly mask some atoms to a special 'mask' token"""
        new_atom_types = atom_types.clone()
        n_nodes = node_mask.sum().item()

        if n_nodes <= 1:
            return atom_types

        # Randomly select atoms to mask
        n_mask = max(1, int(n_nodes * self.config['atom_masking']))
        valid_indices = torch.where(node_mask)[0]
        mask_indices = valid_indices[torch.randperm(len(valid_indices))[:n_mask]]

        # Create mask token (uniform distribution over atom types)
        mask_token = torch.ones_like(atom_types[0]) / atom_types.shape[-1]

        for idx in mask_indices:
            new_atom_types[idx] = mask_token

        return new_atom_types


class MolecularEncoder(nn.Module):
    """
    Molecular encoder for contrastive learning
    Projects molecules to a shared representation space
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode molecular representation

        Args:
            x: [B, input_dim] molecular features

        Returns:
            encoded: [B, output_dim] encoded representations
        """
        encoded = self.encoder(x)
        # L2 normalize for cosine similarity
        return F.normalize(encoded, dim=-1)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for molecular representation learning
    """

    def __init__(self, temperature: float = 0.1, similarity_metric: str = 'cosine'):
        super().__init__()
        self.temperature = temperature
        self.similarity_metric = similarity_metric

    def compute_similarity(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute similarity matrix between two sets of representations"""
        if self.similarity_metric == 'cosine':
            return torch.mm(z1, z2.t()) / self.temperature
        elif self.similarity_metric == 'euclidean':
            # Negative squared Euclidean distance
            return -torch.cdist(z1, z2).pow(2) / self.temperature
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss

        Args:
            z1: [B, D] representations of original molecules
            z2: [B, D] representations of augmented molecules

        Returns:
            loss: Contrastive loss value
        """
        batch_size = z1.shape[0]
        device = z1.device

        # Concatenate representations
        z = torch.cat([z1, z2], dim=0)  # [2B, D]

        # Compute similarity matrix
        similarity_matrix = self.compute_similarity(z, z)  # [2B, 2B]

        # Create positive pairs mask
        mask = torch.zeros(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=device)
        for i in range(batch_size):
            mask[i, batch_size + i] = True  # Original-augmented pairs
            mask[batch_size + i, i] = True  # Augmented-original pairs

        # Remove self-similarity
        similarity_matrix.fill_diagonal_(float('-inf'))

        # Compute positive and negative similarities
        positive_similarity = similarity_matrix[mask]  # Positive pairs
        negative_similarity = similarity_matrix[~mask].view(2 * batch_size, -1)  # Negative pairs

        # InfoNCE loss
        positive_loss = -positive_similarity.mean()

        # For each sample, compute log-sum-exp of negatives
        negative_loss = torch.logsumexp(negative_similarity, dim=1).mean()

        loss = positive_loss + negative_loss

        return loss


class MolecularContrastiveLearner(nn.Module):
    """
    Main contrastive learning module for molecular representation learning
    """

    def __init__(self, base_model, molecular_encoder: MolecularEncoder,
                 augmenter: MolecularAugmenter,
                 contrastive_loss: ContrastiveLoss,
                 contrastive_weight: float = 0.1):
        super().__init__()
        self.base_model = base_model
        self.molecular_encoder = molecular_encoder
        self.augmenter = augmenter
        self.contrastive_loss = contrastive_loss
        self.contrastive_weight = contrastive_weight

    def extract_molecular_representation(self, X: torch.Tensor, E: torch.Tensor,
                                       node_mask: torch.Tensor) -> torch.Tensor:
        """
        Extract molecular-level representation from graph

        Args:
            X: [B, N, atom_dim] atom features
            E: [B, N, N, edge_dim] edge features
            node_mask: [B, N] node mask

        Returns:
            mol_repr: [B, repr_dim] molecular representations
        """
        batch_size = X.shape[0]
        representations = []

        for i in range(batch_size):
            n_nodes = node_mask[i].sum().item()

            if n_nodes == 0:
                # Empty molecule
                mol_repr = torch.zeros(X.shape[-1] + E.shape[-1], device=X.device)
            else:
                # Aggregate node features
                node_repr = X[i, :n_nodes].mean(dim=0)

                # Aggregate edge features
                edge_repr = E[i, :n_nodes, :n_nodes].mean(dim=(0, 1))

                # Combine node and edge representations
                mol_repr = torch.cat([node_repr, edge_repr])

            representations.append(mol_repr)

        return torch.stack(representations)

    def forward(self, X: torch.Tensor, E: torch.Tensor, y: torch.Tensor,
                node_mask: torch.Tensor, compute_contrastive: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional contrastive learning

        Args:
            X: [B, N, atom_dim] atom features
            E: [B, N, N, edge_dim] edge features
            y: [B, y_dim] molecular properties
            node_mask: [B, N] node mask
            compute_contrastive: Whether to compute contrastive loss

        Returns:
            Dictionary containing losses and representations
        """
        results = {}

        if compute_contrastive and self.training:
            batch_size = X.shape[0]

            # Create augmented versions
            X_aug_list, E_aug_list, y_aug_list, node_mask_aug_list = [], [], [], []

            for i in range(batch_size):
                # Extract single molecule
                X_i = X[i]
                E_i = E[i]
                y_i = y[i]
                node_mask_i = node_mask[i]

                # Augment molecule
                X_aug, E_aug, node_mask_aug, y_aug = self.augmenter.augment_molecule(
                    X_i, E_i, node_mask_i, y_i
                )

                X_aug_list.append(X_aug)
                E_aug_list.append(E_aug)
                y_aug_list.append(y_aug)
                node_mask_aug_list.append(node_mask_aug)

            # Stack augmented molecules
            X_aug = torch.stack(X_aug_list)
            E_aug = torch.stack(E_aug_list)
            y_aug = torch.stack(y_aug_list)
            node_mask_aug = torch.stack(node_mask_aug_list)

            # Extract molecular representations
            mol_repr_orig = self.extract_molecular_representation(X, E, node_mask)
            mol_repr_aug = self.extract_molecular_representation(X_aug, E_aug, node_mask_aug)

            # Encode representations
            z_orig = self.molecular_encoder(mol_repr_orig)
            z_aug = self.molecular_encoder(mol_repr_aug)

            # Compute contrastive loss
            contrastive_loss = self.contrastive_loss(z_orig, z_aug)

            results['contrastive_loss'] = contrastive_loss
            results['molecular_representations'] = {
                'original': z_orig,
                'augmented': z_aug
            }

        else:
            results['contrastive_loss'] = torch.tensor(0.0, device=X.device)

        return results

    def compute_total_loss(self, base_loss: torch.Tensor,
                          contrastive_results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Combine base loss with contrastive loss

        Args:
            base_loss: Original model loss
            contrastive_results: Results from contrastive learning

        Returns:
            total_loss: Combined loss
        """
        contrastive_loss = contrastive_results.get('contrastive_loss', torch.tensor(0.0))
        total_loss = base_loss + self.contrastive_weight * contrastive_loss

        return total_loss


class ContrastivePretrainer:
    """
    Pretrainer for molecular contrastive learning
    """

    def __init__(self, model: MolecularContrastiveLearner,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def pretrain_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Run one epoch of contrastive pretraining

        Args:
            dataloader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch_data in enumerate(dataloader):
            # Move data to device
            # Assuming batch_data is a batch of graph data
            X = batch_data.x  # Node features
            E = batch_data.edge_attr  # Edge features
            y = batch_data.y  # Molecular properties

            # Convert to dense representation if needed
            # This would depend on your specific data format

            self.optimizer.zero_grad()

            # Forward pass with contrastive learning
            results = self.model(X, E, y, compute_contrastive=True)

            loss = results['contrastive_loss']
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(num_batches, 1)

        return {
            'contrastive_loss': avg_loss,
            'num_batches': num_batches
        }

    def pretrain(self, dataloader: DataLoader, num_epochs: int):
        """
        Run full contrastive pretraining

        Args:
            dataloader: Training data loader
            num_epochs: Number of pretraining epochs
        """
        print(f"Starting contrastive pretraining for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            metrics = self.pretrain_epoch(dataloader, epoch)
            print(f"Epoch {epoch + 1}/{num_epochs}: "
                  f"Contrastive Loss = {metrics['contrastive_loss']:.4f}")

        print("Contrastive pretraining completed!")