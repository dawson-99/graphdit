"""
Edge-aware attention mechanisms for molecular graph processing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class EdgeAwareAttention(nn.Module):
    """Edge-aware self-attention that incorporates bond information"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        edge_dim: int = 5,  # Number of bond types
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.edge_dim = edge_dim

        # Standard Q, K, V projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Edge-aware components
        self.edge_proj = nn.Linear(edge_dim, num_heads, bias=False)
        self.edge_gate = nn.Linear(edge_dim, num_heads, bias=True)

        # Optional normalization
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

        # Dropout and output projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.edge_proj.weight)
        nn.init.zeros_(self.edge_gate.bias)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_features: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (B, N, D)
            edge_features: Edge features (B, N, N, edge_dim)
            node_mask: Valid node mask (B, N)

        Returns:
            Updated node features (B, N, D)
        """
        B, N, D = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Apply normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Compute attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)

        # Add edge bias
        edge_bias = self.edge_proj(edge_features)  # (B, N, N, num_heads)
        edge_bias = edge_bias.permute(0, 3, 1, 2)  # (B, num_heads, N, N)

        # Edge gating mechanism
        edge_gates = torch.sigmoid(self.edge_gate(edge_features))  # (B, N, N, num_heads)
        edge_gates = edge_gates.permute(0, 3, 1, 2)  # (B, num_heads, N, N)

        # Apply edge-aware bias with gating
        attn_scores = attn_scores + edge_gates * edge_bias

        # Apply node mask
        if node_mask is not None:
            # Create attention mask: valid if both nodes are valid
            attn_mask = node_mask[:, None, :, None] & node_mask[:, None, None, :]
            attn_mask = attn_mask.expand(-1, self.num_heads, -1, -1)

            # Mask invalid positions
            attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))

            # Handle rows with all invalid entries (avoid NaN in softmax)
            row_valid = attn_mask.sum(dim=-1) > 0  # (B, num_heads, N)
            attn_scores = attn_scores.masked_fill(
                ~row_valid.unsqueeze(-1), 0.0
            )

        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # Apply attention to values
        out = attn_weights @ v  # (B, num_heads, N, head_dim)

        # Concatenate heads
        out = out.transpose(1, 2).reshape(B, N, D)  # (B, N, D)

        # Final projection
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class BondTypeAwareAttention(EdgeAwareAttention):
    """Specialized attention that explicitly models different bond types"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Separate embeddings for different bond types
        self.bond_embeddings = nn.Embedding(self.edge_dim, self.num_heads)
        self.bond_position_embeddings = nn.Parameter(
            torch.randn(self.edge_dim, self.head_dim) * 0.02
        )

    def forward(self, x, edge_features, node_mask=None):
        """Enhanced forward with bond type specific processing"""
        B, N, D = x.shape

        # Get bond types (argmax over edge features)
        bond_types = edge_features.argmax(dim=-1)  # (B, N, N)

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Bond-specific key modification
        bond_pos_emb = self.bond_position_embeddings[bond_types]  # (B, N, N, head_dim)
        bond_pos_emb = bond_pos_emb.unsqueeze(1).expand(-1, self.num_heads, -1, -1, -1)

        # Modify keys with bond information
        k_expanded = k.unsqueeze(2).expand(-1, -1, N, -1, -1)  # (B, heads, N, N, head_dim)
        k_bond = k_expanded + bond_pos_emb

        # Compute attention with bond-aware keys
        attn_scores = torch.einsum('bhid,bhijd->bhij', q, k_bond) * self.scale

        # Add bond type bias
        bond_bias = self.bond_embeddings(bond_types)  # (B, N, N, num_heads)
        bond_bias = bond_bias.permute(0, 3, 1, 2)
        attn_scores = attn_scores + bond_bias

        # Apply masks and softmax
        if node_mask is not None:
            attn_mask = node_mask[:, None, :, None] & node_mask[:, None, None, :]
            attn_mask = attn_mask.expand(-1, self.num_heads, -1, -1)
            attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))

            row_valid = attn_mask.sum(dim=-1) > 0
            attn_scores = attn_scores.masked_fill(~row_valid.unsqueeze(-1), 0.0)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # Apply attention
        out = attn_weights @ v
        out = out.transpose(1, 2).reshape(B, N, D)

        # Final projection
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class MultiScaleGraphAttention(nn.Module):
    """Multi-scale attention that captures both local and global molecular patterns"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        edge_dim: int = 5,
        scales: list = [1, 2, 3],  # Different attention scales
        **kwargs
    ):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)

        # Separate attention for each scale
        self.attentions = nn.ModuleList([
            EdgeAwareAttention(
                dim=dim // self.num_scales,
                num_heads=num_heads // self.num_scales,
                edge_dim=edge_dim,
                **kwargs
            ) for _ in scales
        ])

        # Scale-wise projections
        self.scale_projs = nn.ModuleList([
            nn.Linear(dim, dim // self.num_scales) for _ in scales
        ])

        # Output fusion
        self.output_proj = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, edge_features, node_mask=None):
        """Multi-scale attention forward pass"""
        scale_outputs = []

        for i, (scale, attn, proj) in enumerate(zip(self.scales, self.attentions, self.scale_projs)):
            # Project to scale-specific subspace
            x_scale = proj(x)

            # Apply scale-specific processing (could downsample/upsample here)
            x_out = attn(x_scale, edge_features, node_mask)
            scale_outputs.append(x_out)

        # Concatenate multi-scale outputs
        multi_scale_out = torch.cat(scale_outputs, dim=-1)

        # Final projection and residual
        out = self.output_proj(multi_scale_out)
        out = self.layer_norm(out + x)

        return out