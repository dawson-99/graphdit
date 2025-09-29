"""
Dynamic Attention Sparsity for Graph-DiT
Chemical distance-based learnable attention masks for improved efficiency
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from torch.jit import Final


class ChemicalDistanceComputer:
    """
    Computes chemical distances between atoms in molecular graphs
    """

    def __init__(self):
        # Chemical distance weights for different bond types
        self.bond_weights = {
            0: float('inf'),  # No bond
            1: 1.0,          # Single bond
            2: 0.8,          # Double bond (stronger connection)
            3: 0.6,          # Triple bond (strongest connection)
            4: 0.9,          # Aromatic bond
        }

    def compute_chemical_distance_matrix(self, edge_matrix: torch.Tensor,
                                       node_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute chemical distance matrix using shortest path with bond type weighting

        Args:
            edge_matrix: [B, N, N, E] edge features
            node_mask: [B, N] node mask

        Returns:
            distance_matrix: [B, N, N] chemical distances
        """
        batch_size, num_nodes, _, num_edge_types = edge_matrix.shape
        device = edge_matrix.device

        distance_matrices = torch.full(
            (batch_size, num_nodes, num_nodes),
            float('inf'), device=device
        )

        for b in range(batch_size):
            # Create adjacency matrix with weights
            adj_weights = torch.full((num_nodes, num_nodes), float('inf'), device=device)

            for i in range(num_nodes):
                for j in range(num_nodes):
                    if not (node_mask[b, i] and node_mask[b, j]):
                        continue

                    if i == j:
                        adj_weights[i, j] = 0.0
                        continue

                    # Find strongest bond type between i and j
                    bond_probs = edge_matrix[b, i, j]
                    strongest_bond = torch.argmax(bond_probs).item()

                    if strongest_bond in self.bond_weights:
                        weight = self.bond_weights[strongest_bond]
                        if weight != float('inf'):  # Valid bond
                            adj_weights[i, j] = weight

            # Floyd-Warshall algorithm for shortest paths
            dist_matrix = adj_weights.clone()

            for k in range(num_nodes):
                if not node_mask[b, k]:
                    continue
                for i in range(num_nodes):
                    if not node_mask[b, i]:
                        continue
                    for j in range(num_nodes):
                        if not node_mask[b, j]:
                            continue

                        new_dist = dist_matrix[i, k] + dist_matrix[k, j]
                        dist_matrix[i, j] = torch.min(dist_matrix[i, j], new_dist)

            distance_matrices[b] = dist_matrix

        return distance_matrices


class LearnableSparsityPredictor(nn.Module):
    """
    Learns to predict attention sparsity patterns based on molecular features
    """

    def __init__(self, hidden_dim: int, num_heads: int,
                 max_chemical_distance: float = 5.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_chemical_distance = max_chemical_distance

        # Feature extractors for sparsity prediction
        self.node_feature_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.edge_feature_proj = nn.Linear(hidden_dim, hidden_dim // 4)

        # Sparsity prediction network
        self.sparsity_mlp = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim // 4 + 1, hidden_dim // 2),  # +1 for distance
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_heads),
            nn.Sigmoid()  # Output probability of keeping attention
        )

        # Learnable distance embedding
        self.distance_embedding = nn.Embedding(int(max_chemical_distance * 2 + 1), hidden_dim // 4)

    def forward(self, node_features: torch.Tensor,
                chemical_distances: torch.Tensor,
                node_mask: torch.Tensor) -> torch.Tensor:
        """
        Predict attention sparsity masks

        Args:
            node_features: [B, N, D] node features
            chemical_distances: [B, N, N] chemical distance matrix
            node_mask: [B, N] node mask

        Returns:
            sparsity_mask: [B, num_heads, N, N] attention mask probabilities
        """
        batch_size, num_nodes, hidden_dim = node_features.shape

        # Project node features
        node_proj = self.node_feature_proj(node_features)  # [B, N, D//2]

        # Compute pairwise features
        sparsity_probs = torch.zeros(
            batch_size, self.num_heads, num_nodes, num_nodes,
            device=node_features.device
        )

        for b in range(batch_size):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if not (node_mask[b, i] and node_mask[b, j]):
                        continue

                    # Node pair features
                    node_i = node_proj[b, i]  # [D//2]
                    node_j = node_proj[b, j]  # [D//2]

                    # Chemical distance feature
                    chem_dist = chemical_distances[b, i, j]
                    chem_dist_clamped = torch.clamp(
                        chem_dist, 0, self.max_chemical_distance
                    )

                    # Distance embedding
                    dist_idx = torch.round(chem_dist_clamped * 2).long()
                    dist_emb = self.distance_embedding(dist_idx)  # [D//4]

                    # Edge features (interaction between nodes)
                    edge_feat = torch.tanh(node_i + node_j)  # Simple interaction
                    edge_proj = self.edge_feature_proj(
                        torch.cat([edge_feat, torch.zeros(hidden_dim - edge_feat.shape[0],
                                                         device=edge_feat.device)])[:hidden_dim]
                    )[:hidden_dim // 4]

                    # Combined features
                    combined_feat = torch.cat([
                        edge_feat[:hidden_dim // 2],
                        edge_proj,
                        chem_dist_clamped.unsqueeze(0) / self.max_chemical_distance
                    ])

                    # Predict sparsity for each head
                    head_probs = self.sparsity_mlp(combined_feat)  # [num_heads]
                    sparsity_probs[b, :, i, j] = head_probs

        return sparsity_probs


class DynamicSparseAttention(nn.Module):
    """
    Sparse attention mechanism with dynamic, learnable sparsity patterns
    """

    fast_attn: Final[bool]

    def __init__(self, dim: int, num_heads: int = 8,
                 qkv_bias: bool = False, qk_norm: bool = False,
                 attn_drop: float = 0., proj_drop: float = 0.,
                 sparsity_ratio: float = 0.3,
                 chemical_distance_threshold: float = 3.0,
                 use_learnable_sparsity: bool = True,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sparsity_ratio = sparsity_ratio
        self.chemical_distance_threshold = chemical_distance_threshold
        self.use_learnable_sparsity = use_learnable_sparsity

        self.fast_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        # Standard attention components
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Chemical distance computer
        self.distance_computer = ChemicalDistanceComputer()

        # Learnable sparsity predictor
        if use_learnable_sparsity:
            self.sparsity_predictor = LearnableSparsityPredictor(
                dim, num_heads, chemical_distance_threshold
            )

        # Efficiency tracking
        self.sparsity_stats = {
            'total_attention_ops': 0,
            'sparse_attention_ops': 0,
            'sparsity_ratios': []
        }

    def compute_sparsity_mask(self, x: torch.Tensor, edge_features: torch.Tensor,
                            node_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamic sparsity mask based on chemical structure
        """
        batch_size, num_nodes = node_mask.shape

        if self.use_learnable_sparsity and hasattr(self, 'sparsity_predictor'):
            # Learnable sparsity prediction
            chemical_distances = self.distance_computer.compute_chemical_distance_matrix(
                edge_features, node_mask
            )

            sparsity_mask = self.sparsity_predictor(
                x, chemical_distances, node_mask
            )
        else:
            # Rule-based sparsity
            chemical_distances = self.distance_computer.compute_chemical_distance_matrix(
                edge_features, node_mask
            )

            # Create base mask from chemical distance
            distance_mask = chemical_distances <= self.chemical_distance_threshold

            # Expand to multi-head format
            sparsity_mask = distance_mask.unsqueeze(1).expand(
                batch_size, self.num_heads, num_nodes, num_nodes
            ).float()

        # Apply node mask
        node_pair_mask = (
            node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        ).unsqueeze(1).expand(batch_size, self.num_heads, num_nodes, num_nodes)

        sparsity_mask = sparsity_mask * node_pair_mask.float()

        # Ensure diagonal attention (self-attention always allowed)
        diag_mask = torch.eye(num_nodes, device=x.device).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(1).expand(
            batch_size, self.num_heads, num_nodes, num_nodes
        )
        sparsity_mask = torch.where(diag_mask, torch.ones_like(sparsity_mask), sparsity_mask)

        return sparsity_mask

    def apply_top_k_sparsity(self, attention_weights: torch.Tensor,
                           sparsity_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply top-k sparsity to attention weights
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape

        # Apply learned sparsity mask
        masked_attention = attention_weights * sparsity_mask

        # Additional top-k sparsity for efficiency
        k = max(1, int(seq_len * (1 - self.sparsity_ratio)))

        # Find top-k values for each query
        top_k_values, top_k_indices = torch.topk(
            masked_attention, k=k, dim=-1
        )

        # Create sparse attention matrix
        sparse_attention = torch.zeros_like(attention_weights)
        batch_indices = torch.arange(batch_size).view(-1, 1, 1, 1)
        head_indices = torch.arange(num_heads).view(1, -1, 1, 1)
        seq_indices = torch.arange(seq_len).view(1, 1, -1, 1)

        sparse_attention[
            batch_indices, head_indices, seq_indices, top_k_indices
        ] = top_k_values

        return sparse_attention

    def forward(self, x: torch.Tensor, node_mask: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with dynamic sparse attention
        """
        B, N, D = x.shape

        # Generate QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, num_heads, N, head_dim]
        q, k = self.q_norm(q), self.k_norm(k)

        # Compute attention weights
        q_scaled = q * self.scale
        attention_weights = q_scaled @ k.transpose(-2, -1)  # [B, num_heads, N, N]

        # Compute sparsity mask
        if edge_features is not None:
            sparsity_mask = self.compute_sparsity_mask(x, edge_features, node_mask)
        else:
            # Fallback: use distance-based mask without edge features
            sparsity_mask = torch.ones_like(attention_weights)

        # Apply sparsity
        sparse_attention_weights = self.apply_top_k_sparsity(
            attention_weights, sparsity_mask
        )

        # Apply softmax to sparse attention
        sparse_attention_weights = sparse_attention_weights.softmax(dim=-1)
        sparse_attention_weights = self.attn_drop(sparse_attention_weights)

        # Compute output
        out = sparse_attention_weights @ v  # [B, num_heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, D)

        # Final projection
        out = self.proj(out)
        out = self.proj_drop(out)

        # Update efficiency statistics
        total_ops = B * self.num_heads * N * N
        sparse_ops = (sparse_attention_weights > 0).sum().item()
        current_sparsity = 1 - (sparse_ops / total_ops)

        self.sparsity_stats['total_attention_ops'] += total_ops
        self.sparsity_stats['sparse_attention_ops'] += sparse_ops
        self.sparsity_stats['sparsity_ratios'].append(current_sparsity)

        return out

    def get_efficiency_stats(self) -> dict:
        """Get efficiency statistics"""
        total_ops = self.sparsity_stats['total_attention_ops']
        sparse_ops = self.sparsity_stats['sparse_attention_ops']

        if total_ops > 0:
            overall_sparsity = 1 - (sparse_ops / total_ops)
            efficiency_gain = total_ops / max(sparse_ops, 1)
        else:
            overall_sparsity = 0.0
            efficiency_gain = 1.0

        return {
            'overall_sparsity': overall_sparsity,
            'efficiency_gain': efficiency_gain,
            'avg_sparsity_ratio': np.mean(self.sparsity_stats['sparsity_ratios']) if self.sparsity_stats['sparsity_ratios'] else 0.0,
            'total_operations': total_ops,
            'sparse_operations': sparse_ops
        }

    def reset_stats(self):
        """Reset efficiency statistics"""
        self.sparsity_stats = {
            'total_attention_ops': 0,
            'sparse_attention_ops': 0,
            'sparsity_ratios': []
        }


class AdaptiveSparsityScheduler:
    """
    Schedules sparsity ratio during training
    """

    def __init__(self, initial_sparsity: float = 0.1,
                 final_sparsity: float = 0.5,
                 warmup_epochs: int = 10,
                 total_epochs: int = 100):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def get_sparsity_ratio(self, epoch: int) -> float:
        """Get sparsity ratio for current epoch"""
        if epoch < self.warmup_epochs:
            # Linear warmup
            progress = epoch / self.warmup_epochs
            return self.initial_sparsity * (1 - progress) + progress * 0.2
        else:
            # Gradual increase to final sparsity
            remaining_epochs = self.total_epochs - self.warmup_epochs
            progress = min((epoch - self.warmup_epochs) / remaining_epochs, 1.0)

            # Cosine annealing to final sparsity
            sparsity = 0.2 + (self.final_sparsity - 0.2) * (
                1 - np.cos(progress * np.pi)
            ) / 2

            return sparsity

    def update_model_sparsity(self, model, epoch: int):
        """Update sparsity ratio in all sparse attention layers"""
        target_sparsity = self.get_sparsity_ratio(epoch)

        for module in model.modules():
            if isinstance(module, DynamicSparseAttention):
                module.sparsity_ratio = target_sparsity