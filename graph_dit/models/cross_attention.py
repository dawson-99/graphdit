"""
Multi-conditional cross-attention mechanisms for precise condition control
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
import math


class MultiConditionCrossAttention(nn.Module):
    """Cross-attention module that handles multiple types of conditions"""

    def __init__(
        self,
        dim: int,
        condition_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.condition_dim = condition_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query from molecular features
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # Key and Value from conditions
        self.kv_proj = nn.Linear(condition_dim, dim * 2, bias=qkv_bias)

        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, conditions: torch.Tensor, condition_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: Molecular features (B, N, dim)
            conditions: Condition features (B, num_conditions, condition_dim)
            condition_mask: Valid condition mask (B, num_conditions)

        Returns:
            Updated molecular features (B, N, dim)
        """
        B, N, D = x.shape
        _, num_cond, _ = conditions.shape

        # Generate Q, K, V
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(conditions).reshape(B, num_cond, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        # Attention computation
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, num_cond)

        # Apply condition mask
        if condition_mask is not None:
            condition_mask = condition_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, num_cond)
            attn_scores = attn_scores.masked_fill(~condition_mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # Apply attention
        out = attn_weights @ v  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)

        # Final projection
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class HierarchicalConditionProcessor(nn.Module):
    """Processes different types of conditions hierarchically"""

    def __init__(
        self,
        molecular_dim: int = 1152,
        time_dim: int = 256,
        property_dim: int = 64,
        categorical_dims: Dict[str, int] = None,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.molecular_dim = molecular_dim
        self.hidden_dim = hidden_dim

        if categorical_dims is None:
            categorical_dims = {'class': 2, 'source': 3}

        # Time embedding
        self.time_embedder = TimestepEmbedder(time_dim)

        # Property embeddings
        self.property_embedder = nn.Sequential(
            nn.Linear(property_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Categorical embeddings
        self.categorical_embedders = nn.ModuleDict()
        for cat_name, cat_size in categorical_dims.items():
            self.categorical_embedders[cat_name] = nn.Embedding(cat_size, hidden_dim // len(categorical_dims))

        # Condition fusion layers
        total_condition_dim = time_dim + hidden_dim + sum(hidden_dim // len(categorical_dims) for _ in categorical_dims)
        self.condition_fusion = nn.Sequential(
            nn.Linear(total_condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Multi-level cross attentions
        self.cross_attentions = nn.ModuleList([
            MultiConditionCrossAttention(molecular_dim, hidden_dim)
            for _ in range(3)  # Three levels of condition processing
        ])

        # Condition type embeddings (learnable)
        self.condition_type_embeddings = nn.Parameter(
            torch.randn(4, hidden_dim) * 0.02  # time, property, categorical, global
        )

    def forward(
        self,
        x: torch.Tensor,  # Molecular features
        t: torch.Tensor,  # Time steps
        properties: torch.Tensor,  # Continuous properties
        categories: Dict[str, torch.Tensor],  # Categorical conditions
        node_mask: Optional[torch.Tensor] = None,
        condition_weights: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: Molecular features (B, N, molecular_dim)
            t: Time steps (B,)
            properties: Property values (B, property_dim)
            categories: Dictionary of categorical conditions
            node_mask: Valid node mask (B, N)
            condition_weights: Weights for different condition types (B, 4)

        Returns:
            Updated molecular features (B, N, molecular_dim)
        """
        batch_size = x.shape[0]

        # Process different condition types
        conditions = []

        # 1. Time conditions
        time_emb = self.time_embedder(t)  # (B, time_dim)
        time_cond = time_emb + self.condition_type_embeddings[0]
        conditions.append(time_cond)

        # 2. Property conditions
        prop_emb = self.property_embedder(properties)  # (B, hidden_dim)
        prop_cond = prop_emb + self.condition_type_embeddings[1]
        conditions.append(prop_cond)

        # 3. Categorical conditions
        cat_embs = []
        for cat_name, cat_values in categories.items():
            if cat_name in self.categorical_embedders:
                cat_emb = self.categorical_embedders[cat_name](cat_values)
                cat_embs.append(cat_emb)

        if cat_embs:
            cat_cond = torch.cat(cat_embs, dim=-1) + self.condition_type_embeddings[2]
            conditions.append(cat_cond)

        # 4. Global context (learned)
        global_cond = self.condition_type_embeddings[3].unsqueeze(0).expand(batch_size, -1)
        conditions.append(global_cond)

        # Stack conditions
        all_conditions = torch.stack(conditions, dim=1)  # (B, num_conditions, hidden_dim)

        # Fuse conditions
        fused_conditions = self.condition_fusion(all_conditions.view(batch_size, -1))  # (B, hidden_dim)
        fused_conditions = fused_conditions.unsqueeze(1).expand(-1, all_conditions.shape[1], -1)  # (B, num_cond, hidden_dim)

        # Apply condition weights if provided
        if condition_weights is not None:
            condition_weights = condition_weights.unsqueeze(-1)  # (B, num_cond, 1)
            fused_conditions = fused_conditions * condition_weights

        # Multi-level cross attention processing
        out = x
        for cross_attn in self.cross_attentions:
            out_cross = cross_attn(out, fused_conditions)
            out = out + out_cross  # Residual connection

        return out


class AdaptiveConditionWeighting(nn.Module):
    """Learns to weight different conditions based on generation stage and context"""

    def __init__(
        self,
        molecular_dim: int,
        num_condition_types: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(molecular_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Condition weighting network
        self.weight_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for time
            nn.ReLU(),
            nn.Linear(hidden_dim, num_condition_types),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, node_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: Molecular features (B, N, molecular_dim)
            t: Normalized time steps (B, 1)
            node_mask: Valid node mask (B, N)

        Returns:
            Condition weights (B, num_condition_types)
        """
        batch_size = x.shape[0]

        # Aggregate molecular context
        if node_mask is not None:
            x_masked = x * node_mask.unsqueeze(-1).float()
            context = x_masked.sum(dim=1) / node_mask.sum(dim=1, keepdim=True).float()
        else:
            context = x.mean(dim=1)  # (B, molecular_dim)

        # Encode context
        context_emb = self.context_encoder(context)  # (B, hidden_dim)

        # Add time information
        context_with_time = torch.cat([context_emb, t], dim=-1)  # (B, hidden_dim + 1)

        # Predict condition weights
        weights = self.weight_predictor(context_with_time)  # (B, num_condition_types)

        return weights


class TimestepEmbedder(nn.Module):
    """Enhanced timestep embedder for cross-attention"""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings"""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t.view(-1)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # Ensure t_freq is on the same device as the MLP parameters
        if hasattr(self.mlp[0], 'weight'):
            t_freq = t_freq.to(self.mlp[0].weight.device)
        t_emb = self.mlp(t_freq)
        return t_emb