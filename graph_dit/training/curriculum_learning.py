"""
Chemical Complexity Curriculum Learning for Graph-DiT
Progressive training from simple to complex molecules
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, rdmolops
import networkx as nx


class MolecularComplexityScorer:
    """
    Computes molecular complexity scores based on multiple chemical features
    """

    def __init__(self):
        # Weight factors for different complexity components
        self.weights = {
            'ring_complexity': 0.3,
            'functional_groups': 0.2,
            'molecular_size': 0.15,
            'bond_diversity': 0.15,
            'stereochemistry': 0.1,
            'aromaticity': 0.1
        }

    def compute_ring_complexity(self, mol) -> float:
        """Compute complexity based on ring systems"""
        if mol is None:
            return 0.0

        # Count different types of rings
        ring_info = mol.GetRingInfo()
        num_rings = ring_info.NumRings()

        # Penalty for larger rings and fused ring systems
        ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
        if not ring_sizes:
            return 0.0

        # Ring complexity components
        max_ring_size = max(ring_sizes) if ring_sizes else 0
        fused_rings = len([ring for ring in ring_info.AtomRings() if len(ring) > 6])
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

        complexity = (
            num_rings * 0.5 +
            max_ring_size * 0.3 +
            fused_rings * 0.8 +
            aromatic_rings * 0.4
        )

        return min(complexity / 10.0, 1.0)  # Normalize to 0-1

    def compute_functional_group_complexity(self, mol) -> float:
        """Compute complexity based on functional groups"""
        if mol is None:
            return 0.0

        # Common functional group patterns (SMARTS)
        functional_groups = {
            'carbonyl': '[CX3]=[OX1]',
            'carboxyl': '[CX3](=O)[OX2H1]',
            'ester': '[CX3](=O)[OX2H0]',
            'amide': '[CX3](=[OX1])[NX3]',
            'amine': '[NX3;H2,H1,H0;!$(NC=O)]',
            'alcohol': '[OX2H]',
            'ether': '[OD2]([#6])[#6]',
            'aldehyde': '[CX3H1](=O)',
            'ketone': '[CX3](=[OX1])([#6])[#6]',
            'halogen': '[F,Cl,Br,I]',
            'nitro': '[N+](=O)[O-]',
            'sulfur': '[SX2,SX4]',
        }

        complexity = 0.0
        for fg_name, pattern in functional_groups.items():
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
            complexity += len(matches) * 0.2

        return min(complexity, 1.0)

    def compute_molecular_size_complexity(self, mol) -> float:
        """Complexity based on molecular size"""
        if mol is None:
            return 0.0

        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()

        # Size complexity increases non-linearly
        size_complexity = np.log(num_atoms + 1) / np.log(50)  # Normalize to ~50 atoms
        bond_complexity = np.log(num_bonds + 1) / np.log(60)  # Normalize to ~60 bonds

        return min((size_complexity + bond_complexity) / 2, 1.0)

    def compute_bond_diversity(self, mol) -> float:
        """Complexity based on bond type diversity"""
        if mol is None:
            return 0.0

        bond_types = set()
        for bond in mol.GetBonds():
            bond_types.add(bond.GetBondType())

        # More bond types = higher complexity
        return min(len(bond_types) / 4.0, 1.0)  # Max 4 bond types typically

    def compute_stereochemistry_complexity(self, mol) -> float:
        """Complexity based on stereochemistry"""
        if mol is None:
            return 0.0

        # Count chiral centers and double bond stereochemistry
        chiral_centers = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
        unspecified_chiral = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)

        return min((chiral_centers + unspecified_chiral * 0.5) / 5.0, 1.0)

    def compute_aromaticity_complexity(self, mol) -> float:
        """Complexity based on aromatic systems"""
        if mol is None:
            return 0.0

        aromatic_atoms = sum([atom.GetIsAromatic() for atom in mol.GetAtoms()])
        total_atoms = mol.GetNumAtoms()

        if total_atoms == 0:
            return 0.0

        return min(aromatic_atoms / total_atoms, 1.0)

    def compute_overall_complexity(self, mol) -> float:
        """Compute overall molecular complexity score (0-1)"""
        if mol is None:
            return 0.0

        components = {
            'ring_complexity': self.compute_ring_complexity(mol),
            'functional_groups': self.compute_functional_group_complexity(mol),
            'molecular_size': self.compute_molecular_size_complexity(mol),
            'bond_diversity': self.compute_bond_diversity(mol),
            'stereochemistry': self.compute_stereochemistry_complexity(mol),
            'aromaticity': self.compute_aromaticity_complexity(mol)
        }

        # Weighted sum
        overall_score = sum(
            self.weights[component] * score
            for component, score in components.items()
        )

        return min(max(overall_score, 0.0), 1.0)

    def compute_graph_complexity(self, atom_types: torch.Tensor, edge_types: torch.Tensor) -> float:
        """
        Compute complexity directly from graph representation
        (for cases where SMILES/RDKit mol is not available)
        """
        num_atoms = atom_types.sum().item()
        num_bonds = (edge_types.sum() - edge_types[:, :, 0].sum()).item()  # Exclude no-bond type

        if num_atoms == 0:
            return 0.0

        # Basic structural complexity
        size_complexity = np.log(num_atoms + 1) / np.log(30)
        bond_density = num_bonds / (num_atoms ** 2) if num_atoms > 1 else 0

        # Bond type diversity
        bond_type_counts = edge_types.sum(dim=(0, 1))
        non_zero_bonds = (bond_type_counts > 0).sum().item()
        bond_diversity = non_zero_bonds / edge_types.shape[-1]

        # Atom type diversity
        atom_type_counts = atom_types.sum(dim=0)
        non_zero_atoms = (atom_type_counts > 0).sum().item()
        atom_diversity = non_zero_atoms / atom_types.shape[-1]

        # Combined complexity
        complexity = (
            0.3 * size_complexity +
            0.2 * bond_density +
            0.25 * bond_diversity +
            0.25 * atom_diversity
        )

        return min(max(complexity, 0.0), 1.0)


class CurriculumScheduler:
    """
    Manages curriculum learning schedule for molecular complexity
    """

    def __init__(self, total_epochs: int,
                 complexity_stages: List[Tuple[float, float]] = None,
                 warmup_epochs: int = 10):
        """
        Args:
            total_epochs: Total training epochs
            complexity_stages: List of (min_complexity, max_complexity) tuples
            warmup_epochs: Epochs for warmup phase
        """
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

        # Default complexity stages if not provided
        if complexity_stages is None:
            complexity_stages = [
                (0.0, 0.3),  # Simple molecules
                (0.1, 0.5),  # Medium complexity
                (0.3, 0.7),  # Complex molecules
                (0.0, 1.0),  # All molecules
            ]

        self.complexity_stages = complexity_stages
        self.stage_epochs = self._compute_stage_epochs()

    def _compute_stage_epochs(self) -> List[int]:
        """Compute epochs for each complexity stage"""
        remaining_epochs = self.total_epochs - self.warmup_epochs
        epochs_per_stage = remaining_epochs // len(self.complexity_stages)

        stage_epochs = [epochs_per_stage] * len(self.complexity_stages)

        # Distribute remaining epochs
        remaining = remaining_epochs % len(self.complexity_stages)
        for i in range(remaining):
            stage_epochs[-(i+1)] += 1

        return stage_epochs

    def get_complexity_range(self, epoch: int) -> Tuple[float, float]:
        """Get complexity range for current epoch"""
        if epoch < self.warmup_epochs:
            # Warmup: start with simplest molecules
            return (0.0, 0.2)

        adjusted_epoch = epoch - self.warmup_epochs
        cumulative_epochs = 0

        for i, (stage_length, (min_comp, max_comp)) in enumerate(
            zip(self.stage_epochs, self.complexity_stages)
        ):
            if adjusted_epoch < cumulative_epochs + stage_length:
                return (min_comp, max_comp)
            cumulative_epochs += stage_length

        # Fallback to final stage
        return self.complexity_stages[-1]

    def get_current_stage(self, epoch: int) -> int:
        """Get current curriculum stage"""
        if epoch < self.warmup_epochs:
            return 0

        adjusted_epoch = epoch - self.warmup_epochs
        cumulative_epochs = 0

        for i, stage_length in enumerate(self.stage_epochs):
            if adjusted_epoch < cumulative_epochs + stage_length:
                return i + 1
            cumulative_epochs += stage_length

        return len(self.complexity_stages)


class CurriculumDataSampler:
    """
    Samples training data based on curriculum learning schedule
    """

    def __init__(self, complexity_scorer: MolecularComplexityScorer,
                 curriculum_scheduler: CurriculumScheduler):
        self.complexity_scorer = complexity_scorer
        self.curriculum_scheduler = curriculum_scheduler
        self.molecule_complexities = {}  # Cache for computed complexities

    def compute_dataset_complexities(self, dataset) -> Dict[int, float]:
        """Pre-compute complexity scores for entire dataset"""
        complexities = {}

        for idx, data in enumerate(dataset):
            try:
                # Compute complexity from graph representation
                atom_types = data.x  # Assuming one-hot encoding
                edge_attr = data.edge_attr
                edge_index = data.edge_index

                # Convert to adjacency matrix format for complexity computation
                num_nodes = atom_types.shape[0]
                adj_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.long)
                for i, (src, dst) in enumerate(edge_index.T):
                    adj_matrix[src, dst] = edge_attr[i]

                # One-hot encode for complexity computation
                atom_types_oh = torch.eye(atom_types.max() + 1)[atom_types]
                edge_types_oh = torch.eye(edge_attr.max() + 1)[adj_matrix]

                complexity = self.complexity_scorer.compute_graph_complexity(
                    atom_types_oh, edge_types_oh
                )
                complexities[idx] = complexity

            except Exception as e:
                # Fallback complexity if computation fails
                complexities[idx] = 0.5

        self.molecule_complexities = complexities
        return complexities

    def get_curriculum_indices(self, epoch: int, dataset_size: int) -> List[int]:
        """Get indices of molecules that match current curriculum stage"""
        min_complexity, max_complexity = self.curriculum_scheduler.get_complexity_range(epoch)

        valid_indices = []
        for idx in range(dataset_size):
            complexity = self.molecule_complexities.get(idx, 0.5)
            if min_complexity <= complexity <= max_complexity:
                valid_indices.append(idx)

        # Ensure we have enough samples
        if len(valid_indices) < dataset_size // 10:
            # If too few samples, expand the range slightly
            expanded_range = (max(0.0, min_complexity - 0.1),
                            min(1.0, max_complexity + 0.1))
            valid_indices = []
            for idx in range(dataset_size):
                complexity = self.molecule_complexities.get(idx, 0.5)
                if expanded_range[0] <= complexity <= expanded_range[1]:
                    valid_indices.append(idx)

        return valid_indices


class CurriculumLoss:
    """
    Adaptive loss weighting based on molecular complexity
    """

    def __init__(self, complexity_scorer: MolecularComplexityScorer,
                 base_loss_fn, complexity_weight: float = 0.2):
        self.complexity_scorer = complexity_scorer
        self.base_loss_fn = base_loss_fn
        self.complexity_weight = complexity_weight

    def compute_loss(self, pred_X: torch.Tensor, pred_E: torch.Tensor,
                    true_X: torch.Tensor, true_E: torch.Tensor,
                    node_mask: torch.Tensor, epoch: int) -> torch.Tensor:
        """
        Compute curriculum-aware loss with complexity weighting
        """
        # Base loss computation
        base_loss = self.base_loss_fn(pred_X, pred_E, true_X, true_E, node_mask)

        # Compute complexity for each sample in batch
        batch_size = true_X.shape[0]
        complexity_weights = torch.ones(batch_size, device=true_X.device)

        for i in range(batch_size):
            # Extract molecule
            n_nodes = node_mask[i].sum().item()
            atom_types = true_X[i, :n_nodes]
            edge_types = true_E[i, :n_nodes, :n_nodes]

            # Compute complexity
            complexity = self.complexity_scorer.compute_graph_complexity(
                atom_types, edge_types
            )

            # Adjust weight based on training stage
            # Early training: higher weight for simple molecules
            # Later training: higher weight for complex molecules
            stage_progress = min(epoch / 100, 1.0)  # Assuming ~100 epoch training

            if complexity < 0.5:  # Simple molecules
                complexity_weights[i] = 1.0 + (1.0 - stage_progress) * self.complexity_weight
            else:  # Complex molecules
                complexity_weights[i] = 1.0 + stage_progress * self.complexity_weight

        # Apply complexity weighting
        weighted_loss = base_loss * complexity_weights.view(-1, 1, 1)
        return weighted_loss.mean()


class CurriculumTrainer:
    """
    Main curriculum learning trainer that integrates all components
    """

    def __init__(self, model, optimizer, total_epochs: int):
        self.model = model
        self.optimizer = optimizer

        self.complexity_scorer = MolecularComplexityScorer()
        self.curriculum_scheduler = CurriculumScheduler(total_epochs)
        self.curriculum_sampler = CurriculumDataSampler(
            self.complexity_scorer, self.curriculum_scheduler
        )

        # Track curriculum statistics
        self.curriculum_stats = {
            'epoch_complexities': [],
            'stage_transitions': [],
            'complexity_distributions': []
        }

    def setup_dataset(self, dataset):
        """Pre-compute complexities for the dataset"""
        print("Computing molecular complexities for curriculum learning...")
        complexities = self.curriculum_sampler.compute_dataset_complexities(dataset)

        complexity_values = list(complexities.values())
        print(f"Complexity range: {min(complexity_values):.3f} - {max(complexity_values):.3f}")
        print(f"Mean complexity: {np.mean(complexity_values):.3f}")

        return complexities

    def get_curriculum_batch(self, dataset, epoch: int, batch_size: int):
        """Get curriculum-appropriate batch for current epoch"""
        valid_indices = self.curriculum_sampler.get_curriculum_indices(
            epoch, len(dataset)
        )

        # Sample batch from valid indices
        if len(valid_indices) >= batch_size:
            batch_indices = np.random.choice(valid_indices, batch_size, replace=False)
        else:
            batch_indices = np.random.choice(valid_indices, batch_size, replace=True)

        # Record curriculum statistics
        batch_complexities = [
            self.curriculum_sampler.molecule_complexities[idx]
            for idx in batch_indices
        ]

        self.curriculum_stats['epoch_complexities'].append({
            'epoch': epoch,
            'mean_complexity': np.mean(batch_complexities),
            'std_complexity': np.std(batch_complexities),
            'min_complexity': np.min(batch_complexities),
            'max_complexity': np.max(batch_complexities)
        })

        return [dataset[idx] for idx in batch_indices]

    def log_curriculum_progress(self, epoch: int):
        """Log curriculum learning progress"""
        current_stage = self.curriculum_scheduler.get_current_stage(epoch)
        complexity_range = self.curriculum_scheduler.get_complexity_range(epoch)

        print(f"Epoch {epoch}: Curriculum Stage {current_stage}, "
              f"Complexity Range: {complexity_range[0]:.2f} - {complexity_range[1]:.2f}")

        if epoch in self.curriculum_stats['epoch_complexities']:
            stats = self.curriculum_stats['epoch_complexities'][-1]
            print(f"  Batch Complexity: {stats['mean_complexity']:.3f} ± {stats['std_complexity']:.3f}")