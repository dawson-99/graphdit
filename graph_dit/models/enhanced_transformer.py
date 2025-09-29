"""
Enhanced Transformer with all new innovations integrated
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from models.graph_attention import EdgeAwareAttention, BondTypeAwareAttention
from models.cross_attention import (
    HierarchicalConditionProcessor,
    AdaptiveConditionWeighting,
    MultiConditionCrossAttention
)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class EnhancedDenoiser(nn.Module):
    """Enhanced denoiser with edge-aware attention and cross-attention conditions"""

    def __init__(
        self,
        max_n_nodes,
        hidden_size=1152,
        depth=6,
        num_heads=16,
        mlp_ratio=4.0,
        drop_condition=0.1,
        Xdim=118,
        Edim=5,
        ydim=3,
        task_type='regression',
        use_edge_aware_attention=True,
        use_cross_attention=True,
        use_adaptive_weighting=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.ydim = ydim
        self.use_edge_aware_attention = use_edge_aware_attention
        self.use_cross_attention = use_cross_attention
        self.use_adaptive_weighting = use_adaptive_weighting

        # Input embedding
        self.x_embedder = nn.Linear(Xdim + max_n_nodes * Edim, hidden_size, bias=False)

        # Condition processing
        if use_cross_attention:
            self.condition_processor = HierarchicalConditionProcessor(
                molecular_dim=hidden_size,
                property_dim=ydim,
                categorical_dims={'class': 2} if task_type != 'regression' else {}
            )

        if use_adaptive_weighting:
            self.adaptive_weighter = AdaptiveConditionWeighting(
                molecular_dim=hidden_size,
                num_condition_types=4
            )

        # Enhanced transformer layers
        self.encoders = nn.ModuleList([
            EnhancedSELayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                edge_dim=Edim,
                use_edge_aware=use_edge_aware_attention,
                use_cross_attention=use_cross_attention,
            ) for _ in range(depth)
        ])

        # Output layer
        self.out_layer = EnhancedOutLayer(
            max_n_nodes=max_n_nodes,
            hidden_size=hidden_size,
            atom_type=Xdim,
            bond_type=Edim,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
        )

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize transformer layers"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        def _constant_init(module, val):
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, val)
                if module.bias is not None:
                    nn.init.constant_(module.bias, val)

        self.apply(_basic_init)

        # Initialize modulation layers to zero for stable training
        for block in self.encoders:
            if hasattr(block, 'adaLN_modulation'):
                _constant_init(block.adaLN_modulation[-1], 0)

        if hasattr(self.out_layer, 'adaLN_modulation'):
            _constant_init(self.out_layer.adaLN_modulation[-1], 0)

    def forward(self, x, e, node_mask, y, t, unconditioned=False):
        """Enhanced forward pass with all innovations"""
        batch_size, n, _ = x.size()

        # Input embedding
        x_flat = torch.cat([x, e.reshape(batch_size, n, -1)], dim=-1)
        x_emb = self.x_embedder(x_flat)  # (B, N, hidden_size)

        # Process conditions
        conditions = None
        condition_weights = None

        if self.use_cross_attention and not unconditioned:
            # Prepare categorical conditions
            categories = {}
            if self.ydim > 2:  # Assume first 2 are continuous, rest categorical
                categories['class'] = y[:, 2:].argmax(dim=-1) if y.shape[1] > 2 else torch.zeros(batch_size, device=y.device, dtype=torch.long)

            # Process conditions hierarchically
            conditions = {
                't': t.squeeze(-1) if t.dim() > 1 else t,
                'properties': y[:, :2] if y.shape[1] >= 2 else y,
                'categories': categories
            }

        if self.use_adaptive_weighting and not unconditioned:
            condition_weights = self.adaptive_weighter(x_emb, t, node_mask)

        # Enhanced transformer layers
        for i, layer in enumerate(self.encoders):
            x_emb = layer(
                x=x_emb,
                edge_features=e,
                conditions=conditions,
                condition_weights=condition_weights,
                node_mask=node_mask,
                layer_idx=i
            )

        # Output layer
        X_out, E_out, y_out = self.out_layer(x_emb, x, e, conditions, t, node_mask)

        return utils.PlaceHolder(X=X_out, E=E_out, y=y_out).mask(node_mask)


class EnhancedSELayer(nn.Module):
    """Enhanced SE layer with edge-awareness and cross-attention"""

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        edge_dim=5,
        use_edge_aware=True,
        use_cross_attention=True,
        **block_kwargs
    ):
        super().__init__()
        self.use_edge_aware = use_edge_aware
        self.use_cross_attention = use_cross_attention

        # Normalization layers
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

        # Attention mechanisms
        if use_edge_aware:
            self.self_attn = BondTypeAwareAttention(
                dim=hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=True,
                edge_dim=edge_dim,
                **block_kwargs
            )
        else:
            # Fallback to standard attention
            from models.layers import Attention
            self.self_attn = Attention(
                hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, **block_kwargs
            )

        if use_cross_attention:
            self.cross_attn = MultiConditionCrossAttention(
                dim=hidden_size,
                condition_dim=hidden_size,  # Match hidden_size for consistency
                num_heads=num_heads // 2,  # Use fewer heads for cross-attention
            )
            self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False)

        # MLP
        from models.layers import Mlp
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=0.0,
        )

        # Adaptive modulation (replaces AdaLN)
        modulation_dim = 6 if not use_cross_attention else 9  # Extra for cross-attention
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, modulation_dim * hidden_size, bias=True)
        )

    def forward(self, x, edge_features, conditions=None, condition_weights=None,
                node_mask=None, layer_idx=0):
        """Enhanced forward with all attention mechanisms"""

        # Get modulation parameters (simplified for now)
        if conditions is not None and 't' in conditions:
            # Use time embedding for modulation
            from models.cross_attention import TimestepEmbedder
            t_embedder = TimestepEmbedder(x.shape[-1]).to(x.device)
            c = t_embedder(conditions['t'])
        else:
            c = torch.zeros(x.shape[0], x.shape[-1], device=x.device)

        modulation_params = self.adaLN_modulation(c)

        if self.use_cross_attention:
            # 9 parameters: 3 each for self-attn, cross-attn, mlp
            chunks = modulation_params.chunk(9, dim=1)
            (shift_msa, scale_msa, gate_msa,
             shift_cross, scale_cross, gate_cross,
             shift_mlp, scale_mlp, gate_mlp) = chunks
        else:
            # 6 parameters: 3 each for self-attn, mlp
            chunks = modulation_params.chunk(6, dim=1)
            (shift_msa, scale_msa, gate_msa,
             shift_mlp, scale_mlp, gate_mlp) = chunks

        # Self-attention
        if self.use_edge_aware:
            attn_out = self.self_attn(x, edge_features, node_mask)
        else:
            attn_out = self.self_attn(x, node_mask=node_mask)

        x = x + gate_msa.unsqueeze(1) * modulate(
            self.norm1(attn_out), shift_msa, scale_msa
        )

        # Cross-attention with conditions
        if self.use_cross_attention and conditions is not None:
            # Create condition tensor (simplified)
            condition_tensor = c.unsqueeze(1).expand(-1, 4, -1)  # 4 condition types

            cross_out = self.cross_attn(x, condition_tensor)
            x = x + gate_cross.unsqueeze(1) * modulate(
                self.norm_cross(cross_out), shift_cross, scale_cross
            )

        # MLP
        mlp_out = self.mlp(x)
        x = x + gate_mlp.unsqueeze(1) * modulate(
            self.norm2(mlp_out), shift_mlp, scale_mlp
        )

        return x


class EnhancedOutLayer(nn.Module):
    """Enhanced output layer with better condition integration"""

    def __init__(self, max_n_nodes, hidden_size, atom_type, bond_type, mlp_ratio, num_heads=None):
        super().__init__()
        self.atom_type = atom_type
        self.bond_type = bond_type
        final_size = atom_type + max_n_nodes * bond_type

        from models.layers import Mlp
        self.xedecoder = Mlp(
            in_features=hidden_size,
            out_features=final_size,
            drop=0
        )

        self.norm_final = nn.LayerNorm(final_size, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * final_size, bias=True)
        )

    def forward(self, x, x_in, e_in, conditions, t, node_mask):
        """Enhanced output layer forward"""
        # Get condition embedding for modulation
        if conditions is not None and 't' in conditions:
            from models.cross_attention import TimestepEmbedder
            t_embedder = TimestepEmbedder(x.shape[-1]).to(x.device)
            c = t_embedder(conditions['t'])
        else:
            c = torch.zeros(x.shape[0], x.shape[-1], device=x.device)

        # Decode to atom and bond logits
        x_all = self.xedecoder(x)
        B, N, D = x_all.size()

        # Apply modulation
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x_all = modulate(self.norm_final(x_all), shift, scale)

        # Split to atoms and bonds
        atom_out = x_all[:, :, :self.atom_type]
        atom_out = x_in + atom_out  # Residual connection

        bond_out = x_all[:, :, self.atom_type:].reshape(B, N, N, self.bond_type)
        bond_out = e_in + bond_out  # Residual connection

        # Ensure bond matrix symmetry
        edge_mask = (~node_mask)[:, :, None] & (~node_mask)[:, None, :]
        diag_mask = torch.eye(N, dtype=torch.bool, device=x.device).unsqueeze(0).expand(B, -1, -1)

        bond_out.masked_fill_(edge_mask[:, :, :, None], 0)
        bond_out.masked_fill_(diag_mask[:, :, :, None], 0)
        bond_out = 0.5 * (bond_out + torch.transpose(bond_out, 1, 2))

        return atom_out, bond_out, None