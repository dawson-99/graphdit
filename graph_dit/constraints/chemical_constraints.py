"""
Chemical constraint validation and guidance for molecular generation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from enum import Enum


class BondType(Enum):
    """Bond type enumeration"""
    NONE = 0
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    AROMATIC = 4


class AtomType(Enum):
    """Common atom types (simplified)"""
    C = 0   # Carbon
    N = 1   # Nitrogen
    O = 2   # Oxygen
    F = 3   # Fluorine
    P = 4   # Phosphorus
    S = 5   # Sulfur
    Cl = 6  # Chlorine
    Br = 7  # Bromine
    I = 8   # Iodine


class ChemicalConstraintValidator:
    """Validates chemical constraints for molecular graphs"""

    def __init__(self, device='cuda'):
        self.device = device

        # Valence rules (max bonds for each atom type)
        self.max_valences = {
            AtomType.C.value: 4,
            AtomType.N.value: 3,
            AtomType.O.value: 2,
            AtomType.F.value: 1,
            AtomType.P.value: 5,
            AtomType.S.value: 6,
            AtomType.Cl.value: 1,
            AtomType.Br.value: 1,
            AtomType.I.value: 1,
        }

        # Bond type weights for valence calculation
        self.bond_weights = {
            BondType.NONE.value: 0,
            BondType.SINGLE.value: 1,
            BondType.DOUBLE.value: 2,
            BondType.TRIPLE.value: 3,
            BondType.AROMATIC.value: 1.5,
        }

    def validate_valences(self, atom_types: torch.Tensor, bond_matrix: torch.Tensor,
                         node_mask: torch.Tensor) -> torch.Tensor:
        """
        Validate valence constraints

        Args:
            atom_types: Atom type indices (B, N)
            bond_matrix: Bond type indices (B, N, N)
            node_mask: Valid node mask (B, N)

        Returns:
            Valence violation mask (B, N) - True where valence is violated
        """
        batch_size, max_nodes = atom_types.shape

        violations = torch.zeros_like(node_mask, dtype=torch.bool)

        for b in range(batch_size):
            for i in range(max_nodes):
                if not node_mask[b, i]:
                    continue

                atom_type = atom_types[b, i].item()
                if atom_type not in self.max_valences:
                    continue

                # Calculate current valence
                current_valence = 0
                for j in range(max_nodes):
                    if i != j and node_mask[b, j]:
                        bond_type = bond_matrix[b, i, j].item()
                        if bond_type in self.bond_weights:
                            current_valence += self.bond_weights[bond_type]

                # Check valence constraint
                max_valence = self.max_valences[atom_type]
                if current_valence > max_valence:
                    violations[b, i] = True

        return violations

    def validate_connectivity(self, bond_matrix: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """
        Check if the molecule is connected

        Args:
            bond_matrix: Bond adjacency matrix (B, N, N)
            node_mask: Valid node mask (B, N)

        Returns:
            Connectivity mask (B,) - True if connected
        """
        batch_size = bond_matrix.shape[0]
        is_connected = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Create adjacency matrix (any bond > 0)
        adj_matrix = (bond_matrix > 0).float()

        for b in range(batch_size):
            num_nodes = node_mask[b].sum().item()
            if num_nodes <= 1:
                is_connected[b] = True
                continue

            # Floyd-Warshall to check connectivity
            reachable = adj_matrix[b, :num_nodes, :num_nodes].clone()

            # Add self-loops
            for i in range(num_nodes):
                reachable[i, i] = 1

            # Floyd-Warshall
            for k in range(num_nodes):
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        reachable[i, j] = torch.max(reachable[i, j], reachable[i, k] * reachable[k, j])

            # Check if all nodes are reachable from node 0
            is_connected[b] = torch.all(reachable[0, :] > 0)

        return is_connected

    def validate_bond_consistency(self, bond_matrix: torch.Tensor) -> torch.Tensor:
        """
        Ensure bond matrix is symmetric

        Args:
            bond_matrix: Bond matrix (B, N, N)

        Returns:
            Symmetry violation mask (B,) - True if symmetric
        """
        # Check if matrix is symmetric
        is_symmetric = torch.allclose(bond_matrix, bond_matrix.transpose(-1, -2), atol=1e-6)
        return torch.full((bond_matrix.shape[0],), is_symmetric, device=self.device)


class ConstraintGuidedSampler:
    """Sampling with chemical constraint guidance"""

    def __init__(self, base_sampler, constraint_validator, guidance_strength: float = 1.0):
        self.base_sampler = base_sampler
        self.validator = constraint_validator
        self.guidance_strength = guidance_strength

    def sample_with_constraints(self, *args, max_retries: int = 10, **kwargs):
        """
        Sample molecules with constraint enforcement

        Args:
            max_retries: Maximum number of retry attempts
            *args, **kwargs: Arguments for base sampler

        Returns:
            List of valid molecules
        """
        all_molecules = []
        target_count = kwargs.get('batch_size', 1)

        retry_count = 0
        while len(all_molecules) < target_count and retry_count < max_retries:
            # Sample from base model
            candidate_molecules = self.base_sampler.sample_fast(*args, **kwargs)

            # Validate and filter
            valid_molecules = self._filter_valid_molecules(candidate_molecules)
            all_molecules.extend(valid_molecules)

            retry_count += 1

        return all_molecules[:target_count]

    def _filter_valid_molecules(self, molecules: List) -> List:
        """Filter molecules based on chemical constraints"""
        valid_molecules = []

        for mol_data in molecules:
            atom_types, bond_matrix = mol_data
            batch_size = 1
            max_nodes = atom_types.shape[0]

            # Create tensors for validation
            atom_tensor = atom_types.argmax(dim=-1).unsqueeze(0)  # (1, N)
            bond_tensor = bond_matrix.argmax(dim=-1).unsqueeze(0)  # (1, N, N)
            node_mask = (atom_types.sum(dim=-1) > 0).unsqueeze(0)  # (1, N)

            # Validate constraints
            valence_violations = self.validator.validate_valences(atom_tensor, bond_tensor, node_mask)
            is_connected = self.validator.validate_connectivity(bond_tensor, node_mask)
            is_symmetric = self.validator.validate_bond_consistency(bond_tensor)

            # Check if molecule passes all constraints
            if (not valence_violations.any() and
                is_connected.all() and
                is_symmetric.all()):
                valid_molecules.append(mol_data)

        return valid_molecules


class ConstraintGuidedLoss:
    """Loss function that incorporates chemical constraints"""

    def __init__(self, validator: ChemicalConstraintValidator,
                 constraint_weights: Dict[str, float] = None):
        self.validator = validator

        if constraint_weights is None:
            constraint_weights = {
                'valence': 10.0,
                'connectivity': 5.0,
                'symmetry': 1.0,
            }
        self.constraint_weights = constraint_weights

    def compute_constraint_loss(self, pred_X: torch.Tensor, pred_E: torch.Tensor,
                               node_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute constraint violation loss

        Args:
            pred_X: Predicted atom probabilities (B, N, num_atom_types)
            pred_E: Predicted bond probabilities (B, N, N, num_bond_types)
            node_mask: Valid node mask (B, N)

        Returns:
            Constraint loss tensor (B,)
        """
        batch_size = pred_X.shape[0]
        device = pred_X.device

        # Sample discrete states for constraint checking
        atom_types = torch.argmax(pred_X, dim=-1)  # (B, N)
        bond_types = torch.argmax(pred_E, dim=-1)  # (B, N, N)

        total_loss = torch.zeros(batch_size, device=device)

        # Valence constraint loss
        if self.constraint_weights['valence'] > 0:
            valence_violations = self.validator.validate_valences(atom_types, bond_types, node_mask)
            valence_loss = valence_violations.float().sum(dim=-1)  # (B,)
            total_loss += self.constraint_weights['valence'] * valence_loss

        # Connectivity constraint loss
        if self.constraint_weights['connectivity'] > 0:
            is_connected = self.validator.validate_connectivity(bond_types, node_mask)
            connectivity_loss = (~is_connected).float()  # (B,)
            total_loss += self.constraint_weights['connectivity'] * connectivity_loss

        # Symmetry constraint loss
        if self.constraint_weights['symmetry'] > 0:
            # Soft symmetry loss using probabilities
            symmetry_loss = F.mse_loss(pred_E, pred_E.transpose(-2, -3), reduction='none')
            symmetry_loss = symmetry_loss.sum(dim=(-1, -2, -3))  # (B,)
            total_loss += self.constraint_weights['symmetry'] * symmetry_loss

        return total_loss


class AdaptiveConstraintScheduler:
    """Adaptive scheduling of constraint strength during training"""

    def __init__(self, initial_strength: float = 0.1, final_strength: float = 1.0,
                 warmup_steps: int = 1000):
        self.initial_strength = initial_strength
        self.final_strength = final_strength
        self.warmup_steps = warmup_steps

    def get_constraint_strength(self, step: int) -> float:
        """Get current constraint strength based on training step"""
        if step < self.warmup_steps:
            # Linear warmup
            progress = step / self.warmup_steps
            return self.initial_strength + progress * (self.final_strength - self.initial_strength)
        else:
            return self.final_strength

    def update_constraint_weights(self, constraint_loss: ConstraintGuidedLoss, step: int):
        """Update constraint weights based on current step"""
        strength = self.get_constraint_strength(step)

        for key in constraint_loss.constraint_weights:
            constraint_loss.constraint_weights[key] *= strength