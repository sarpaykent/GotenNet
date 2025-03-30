# Standard library imports
from functools import partial
from typing import Callable, List, Mapping, Optional, Tuple, Union

# Related third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter, softmax

# Local application/library specific imports
import gotennet.utils as utils
from gotennet.models.components.layers import (
    MLP,
    CosineCutoff,
    Dense,
    Distance,
    EdgeInit,
    NodeInit,
    TensorInit,
    TensorLayerNorm,
    get_weight_init_by_string,
    str2act,
    str2basis,
)

log = utils.get_logger(__name__)

# num_nodes and hidden_dims are placeholder values, will be overwritten by actual data
num_nodes = hidden_dims = 1


def get_split_sizes_from_lmax(lmax: int, start: int = 1) -> List[int]:
    """
    Return split sizes for torch.split based on lmax.
    
    This function calculates the dimensions of spherical harmonic components
    for each angular momentum value from start to lmax.
    
    Args:
        lmax: Maximum angular momentum value
        start: Starting angular momentum value (default: 1)

    Returns:
        List of split sizes for torch.split (sizes of spherical harmonic components)
    """
    return [2 * l + 1 for l in range(start, lmax + 1)]


def split_to_components(tensor: Tensor, lmax: int, start: int = 1, dim: int = -1) -> List[Tensor]:
    """
    Split a tensor into its spherical harmonic components.
    
    This function splits a tensor containing concatenated spherical harmonic components
    into a list of separate tensors, each corresponding to a specific angular momentum.
    
    Args:
        tensor: The tensor to split [*, sum(2l+1 for l in range(start, lmax+1)), *]
        lmax: Maximum angular momentum value
        start: Starting angular momentum value (default: 1)
        dim: The dimension to split along (default: -1)

    Returns:
        List of tensors, each representing a spherical harmonic component
    """
    split_sizes = get_split_sizes_from_lmax(lmax, start=start)
    components = torch.split(tensor, split_sizes, dim=dim)
    return components


class GATA(MessagePassing):
    def __init__(
        self,
        n_atom_basis: int,
        activation: Callable,
        weight_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        aggr: str = "add",
        node_dim: int = 0,
        epsilon: float = 1e-7,
        layer_norm: str = "",
        steerable_norm: str = "",
        cutoff: float = 5.0,
        num_heads: int = 8,
        dropout: float = 0.0,
        edge_updates: Union[bool, str] = True,
        last_layer: bool = False,
        scale_edge: bool = True,
        evec_dim: Optional[int] = None,
        emlp_dim: Optional[int] = None,
        sep_htr: bool = True,
        sep_dir: bool = True,
        sep_tensor: bool = True,
        lmax: int = 2
    ): 
        """
        Graph Attention Transformer Architecture.
        
        Args:
            n_atom_basis: Number of features to describe atomic environments.
            activation: Activation function to be used. If None, no activation function is used.
            weight_init: Weight initialization function.
            bias_init: Bias initialization function.
            aggr: Aggregation method ('add', 'mean' or 'max').
            node_dim: The axis along which to aggregate.
            epsilon: Small constant for numerical stability.
            layer_norm: Type of layer normalization to use.
            steerable_norm: Type of steerable normalization to use.
            cutoff: Cutoff distance for interactions.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            edge_updates: Whether to update edge features.
            last_layer: Whether this is the last layer.
            scale_edge: Whether to scale edge features.
            evec_dim: Dimension of edge vector features.
            emlp_dim: Dimension of edge MLP features.
            sep_htr: Whether to separate vector features.
            sep_dir: Whether to separate direction features.
            sep_tensor: Whether to separate tensor features.
            lmax: Maximum angular momentum.
        """
        super(GATA, self).__init__(aggr=aggr, node_dim=node_dim)
        self.sep_htr = sep_htr
        self.epsilon = epsilon
        self.last_layer = last_layer
        self.edge_updates = edge_updates
        self.scale_edge = scale_edge
        self.activation = activation
        self.sep_dir = sep_dir
        self.sep_tensor = sep_tensor

        # Parse edge update configuration
        update_info = {
            "gated": False,
            "rej": True,
            "mlp": False,
            "mlpa": False,
            "lin_w": 0,
            "lin_ln": 0,
        }

        update_parts = edge_updates.split("_") if isinstance(edge_updates, str) else []
        allowed_parts = [
            "gated", "gatedt", "norej", "norm", "mlp", "mlpa",
            "act", "linw", "linwa", "ln", "postln"
        ]

        if not all([part in allowed_parts for part in update_parts]):
            raise ValueError(f"Invalid edge update parts. Allowed parts are {allowed_parts}")

        if "gated" in update_parts:
            update_info["gated"] = "gated"
        if "gatedt" in update_parts:
            update_info["gated"] = "gatedt"
        if "act" in update_parts:
            update_info["gated"] = "act"
        if "norej" in update_parts:
            update_info["rej"] = False
        if "mlp" in update_parts:
            update_info["mlp"] = True
        if "mlpa" in update_parts:
            update_info["mlpa"] = True
        if "linw" in update_parts:
            update_info["lin_w"] = 1
        if "linwa" in update_parts:
            update_info["lin_w"] = 2
        if "ln" in update_parts:
            update_info["lin_ln"] = 1
        if "postln" in update_parts:
            update_info["lin_ln"] = 2

        self.update_info = update_info
        log.info(f"Edge updates: {update_info}")

        self.dropout = dropout
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax

        # Calculate multiplier based on configuration
        multiplier = 3
        if self.sep_dir:
            multiplier += lmax - 1
        if self.sep_tensor:
            multiplier += lmax - 1
        self.multiplier = multiplier

        # Initialize layers
        InitDense = partial(Dense, weight_init=weight_init, bias_init=bias_init)

        # Implementation of gamma_s function
        self.gamma_s = nn.Sequential(
            InitDense(n_atom_basis, n_atom_basis, activation=activation),
            InitDense(n_atom_basis, multiplier * n_atom_basis, activation=None),
        )

        self.num_heads = num_heads

        # Query and key transformations
        self.W_q = InitDense(n_atom_basis, n_atom_basis, activation=None)
        self.W_k = InitDense(n_atom_basis, n_atom_basis, activation=None)

        # Value transformation
        self.gamma_v = nn.Sequential(
            InitDense(n_atom_basis, n_atom_basis, activation=activation),
            InitDense(n_atom_basis, multiplier * n_atom_basis, activation=None),
        )

        # Edge feature transformations
        self.W_re = InitDense(
            n_atom_basis,
            n_atom_basis,
            activation=activation,
        )

        # Initialize MLP for edge updates
        InitMLP = partial(MLP, weight_init=weight_init, bias_init=bias_init)

        self.edge_vec_dim = n_atom_basis if evec_dim is None else evec_dim
        self.edge_mlp_dim = n_atom_basis if emlp_dim is None else emlp_dim

        if not self.last_layer and self.edge_updates:
            if self.update_info["mlp"] or self.update_info["mlpa"]:
                dims = [n_atom_basis, self.edge_mlp_dim, n_atom_basis]
            else:
                dims = [n_atom_basis, n_atom_basis]

            self.gamma_t = InitMLP(
                dims,
                activation=activation,
                last_activation=None if self.update_info["mlp"] else self.activation,
                norm=""
            )

            self.W_vq = InitDense(n_atom_basis, self.edge_vec_dim, activation=None, bias=False)

            if self.sep_htr:
                self.W_vk = nn.ModuleList([
                    InitDense(n_atom_basis, self.edge_vec_dim, activation=None, bias=False)
                    for _i in range(self.lmax)
                ])
            else:
                self.W_vk = InitDense(n_atom_basis, self.edge_vec_dim, activation=None, bias=False)

            modules = []
            if self.update_info["lin_w"] > 0:
                if self.update_info["lin_ln"] == 1:
                    modules.append(nn.LayerNorm(self.edge_vec_dim))
                if self.update_info["lin_w"] % 10 == 2:
                    modules.append(self.activation)

                self.W_edp = InitDense(
                    self.edge_vec_dim,
                    n_atom_basis,
                    activation=None,
                    norm="layer" if self.update_info["lin_ln"] == 2 else ""
                )

                modules.append(self.W_edp)

            if self.update_info["gated"] == "gatedt":
                modules.append(nn.Tanh())
            elif self.update_info["gated"] == "gated":
                modules.append(nn.Sigmoid())
            elif self.update_info["gated"] == "act":
                modules.append(nn.SiLU())
            self.gamma_w = nn.Sequential(*modules)


        # Cutoff function
        self.cutoff = CosineCutoff(cutoff)
        self._alpha = None

        # Spatial filter
        self.W_rs = InitDense(
            n_atom_basis,
            n_atom_basis * self.multiplier,
            activation=None,
        )

        # Normalization layers
        self.layernorm_ = layer_norm
        self.steerable_norm_ = steerable_norm
        self.layernorm = nn.LayerNorm(n_atom_basis) if layer_norm != "" else nn.Identity()
        self.tensor_layernorm = (
            TensorLayerNorm(n_atom_basis, trainable=False, lmax=self.lmax)
            if steerable_norm != "" else nn.Identity()
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reset all learnable parameters of the module."""
        if self.layernorm_:
            self.layernorm.reset_parameters()

        if self.steerable_norm_:
            self.tensor_layernorm.reset_parameters()

        for l in self.gamma_s:
            l.reset_parameters()

        self.W_q.reset_parameters()
        self.W_k.reset_parameters()

        for l in self.gamma_v:
            l.reset_parameters()

        self.W_rs.reset_parameters()

        if not self.last_layer and self.edge_updates:
            self.gamma_t.reset_parameters()
            self.W_vq.reset_parameters()

            if self.sep_htr:
                for w in self.W_vk:
                    w.reset_parameters()
            else:
                self.W_vk.reset_parameters()

            if self.update_info["lin_w"] > 0:
                self.W_edp.reset_parameters()

    @staticmethod
    def vector_rejection(rep: Tensor, rl_ij: Tensor) -> Tensor:
        """
        Compute the vector rejection of vec onto rl_ij.
        
        Args:
            rep: Input tensor representation [num_edges, (L_max ** 2) - 1, hidden_dims]
            rl_ij: High-degree steerable feature tensor [num_edges, (L_max ** 2) - 1, 1]
            
        Returns:
            The component of vec orthogonal to rl_ij
        """
        vec_proj = (rep * rl_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return rep - vec_proj * rl_ij.unsqueeze(2)

    def forward(
        self,
        edge_index: Tensor,
        h: Tensor,
        X: Tensor,
        rl_ij: Tensor,
        t_ij: Tensor,
        r_ij: Tensor,
        n_edges: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute interaction output for the GATA layer.

        This method processes node and edge features through the attention mechanism
        and updates both scalar and high-degree steerable features.

        Args:
            edge_index: Tensor describing graph connectivity [2, num_edges]
            h: Scalar input values [num_nodes, 1, hidden_dims]
            X: High-degree steerable features [num_nodes, (L_max ** 2) - 1, hidden_dims]
            rl_ij: Edge tensor representation [num_nodes, (L_max ** 2) - 1, 1]
            t_ij: Edge scalar features [num_nodes, 1, hidden_dims]
            r_ij: Edge scalar distance [num_nodes, 1]
            n_edges: Number of edges per node [num_edges, 1]

        Returns:
            Tuple containing:
                - Updated scalar values [num_nodes, 1, hidden_dims]
                - Updated high-degree steerable features [num_nodes, (L_max ** 2) - 1, hidden_dims]
                - Updated edge features [num_edges, 1, hidden_dims]
        """
        h = self.layernorm(h)
        X = self.tensor_layernorm(X)

        q = self.W_q(h).reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)
        k = self.W_k(h).reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)

        # inter-atomic
        x = self.gamma_s(h)
        v = self.gamma_v(h)
        t_ij_attn = self.W_re(t_ij)
        t_ij_filter = self.W_rs(t_ij)

        # propagate_type: (x: Tensor, q:Tensor, k:Tensor, v:Tensor, X: Tensor,
        #                  t_ij_filter: Tensor, t_ij_attn: Tensor, r_ij: Tensor,
        #                  rl_ij: Tensor, n_edges: Tensor)
        d_h, d_X = self.propagate(
            edge_index=edge_index,
            x=x,
            q=q,
            k=k,
            v=v,
            X=X,
            t_ij_filter=t_ij_filter,
            t_ij_attn=t_ij_attn,
            r_ij=r_ij,
            rl_ij=rl_ij,
            n_edges=n_edges
        )

        h = h + d_h
        X = X + d_X

        if not self.last_layer and self.edge_updates:
            X_htr = X

            EQ = self.W_vq(X_htr)
            if self.sep_htr:
                X_split = torch.split(X_htr, get_split_sizes_from_lmax(self.lmax), dim=1)
                EK = torch.concat([
                    w(X_split[i]) for i, w in enumerate(self.W_vk)
                ], dim=1)
            else:
                EK = self.W_vk(X_htr)

            # edge_updater_type: (EQ: Tensor, EK:Tensor, rl_ij: Tensor, t_ij: Tensor)
            dt_ij = self.edge_updater(edge_index, EQ=EQ, EK=EK, rl_ij=rl_ij, t_ij=t_ij)
            t_ij = t_ij + dt_ij
            self._alpha = None
            return h, X, t_ij

        self._alpha = None
        return h, X, t_ij

    def message(
        self,
        edge_index: Tensor,
        x_j: Tensor,
        q_i: Tensor,
        k_j: Tensor,
        v_j: Tensor,
        X_j: Tensor,
        t_ij_filter: Tensor,
        t_ij_attn: Tensor,
        r_ij: Tensor,
        rl_ij: Tensor,
        n_edges: Tensor,
        index: Tensor,
        ptr: OptTensor,
        dim_size: Optional[int]
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute messages from source nodes to target nodes.
        
        This method implements the message passing mechanism for the GATA layer,
        combining attention-based and spatial filtering approaches.
        
        Args:
            edge_index: Edge connectivity tensor [2, num_edges]
            x_j: Source node features [num_edges, 1, hidden_dims]
            q_i: Target node query features [num_edges, num_heads, hidden_dims // num_heads]
            k_j: Source node key features [num_edges, num_heads, hidden_dims // num_heads]
            v_j: Source node value features [num_edges, num_heads, hidden_dims * multiplier // num_heads]
            X_j: Source node high-degree steerable features [num_edges, (L_max ** 2) - 1, hidden_dims]
            t_ij_filter: Edge scalar filter features [num_edges, 1, hidden_dims]
            t_ij_attn: Edge attention filter features [num_edges, 1, hidden_dims]
            r_ij: Edge scalar distance [num_edges, 1]
            rl_ij: Edge tensor representation [num_edges, (L_max ** 2) - 1, 1]
            n_edges: Number of edges per node [num_edges, 1]
            index: Index tensor for scatter operation
            ptr: Pointer tensor for scatter operation
            dim_size: Dimension size for scatter operation

        Returns:
            Tuple containing:
                - Scalar updates dh [num_edges, 1, hidden_dims]
                - High-degree steerable updates dX [num_edges, (L_max ** 2) - 1, hidden_dims]
        """
        # Reshape attention features
        t_ij_attn = t_ij_attn.reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)

        # Compute attention scores
        attn = (q_i * k_j * t_ij_attn).sum(dim=-1, keepdim=True)
        attn = softmax(attn, index, ptr, dim_size)

        # Normalize the attention scores
        if self.scale_edge:
            norm = torch.sqrt(n_edges.reshape(-1, 1, 1)) / np.sqrt(self.n_atom_basis)
        else:
            norm = 1.0 / np.sqrt(self.n_atom_basis)

        attn = attn * norm
        self._alpha = attn
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # Apply attention to values
        sea_ij = attn * v_j.reshape(-1, self.num_heads, (self.n_atom_basis*self.multiplier) // self.num_heads)
        sea_ij = sea_ij.reshape(-1, 1, self.n_atom_basis*self.multiplier)

        # Apply spatial filter
        spatial_attn = t_ij_filter.unsqueeze(1) * x_j * self.cutoff(r_ij.unsqueeze(-1).unsqueeze(-1))

        # Combine attention and spatial components
        outputs = spatial_attn + sea_ij

        # Split outputs into components
        components = torch.split(outputs, self.n_atom_basis, dim=-1)

        o_s_ij = components[0]
        components = components[1:]

        # Process direction components if enabled
        if self.sep_dir:
            o_d_l_ij, components = components[:self.lmax], components[self.lmax:]
            rl_ij_split = split_to_components(rl_ij[..., None], self.lmax, dim=1)
            dir_comps = [rl_ij_split[i] * o_d_l_ij[i] for i in range(self.lmax)]
            dX_R = torch.cat(dir_comps, dim=1)
        else:
            o_d_ij, components = components[0], components[1:]
            dX_R = o_d_ij * rl_ij[..., None]

        # Process tensor components if enabled
        if self.sep_tensor:
            o_t_l_ij = components[:self.lmax]
            X_j_split = split_to_components(X_j, self.lmax, dim=1)
            tensor_comps = [X_j_split[i] * o_t_l_ij[i] for i in range(self.lmax)]
            dX_X = torch.cat(tensor_comps, dim=1)
        else:
            o_t_ij = components[0]
            dX_X = o_t_ij * X_j

        # Combine components
        dX = dX_R + dX_X
        return o_s_ij, dX

    def edge_update(
        self,
        EQ_i: Tensor,
        EK_j: Tensor,
        rl_ij: Tensor,
        t_ij: Tensor
    ) -> Tensor:
        """
        Update edge features based on node features.
        
        This method computes updates to edge features by combining information from
        source and target nodes' high-degree steerable features, potentially applying
        vector rejection.
        
        Args:
            EQ_i: Source node high-degree steerable features [num_edges, (L_max ** 2) - 1, hidden_dims]
            EK_j: Target node high-degree steerable features [num_edges, (L_max ** 2) - 1, hidden_dims]
            rl_ij: Edge tensor representation [num_edges, (L_max ** 2) - 1, 1]
            t_ij: Edge scalar features [num_edges, 1, hidden_dims]
            
        Returns:
            Updated edge features [num_edges, 1, hidden_dims]
        """
        if self.sep_htr:
            EQ_i_split = split_to_components(EQ_i, self.lmax, dim=1)
            EK_j_split = split_to_components(EK_j, self.lmax, dim=1)
            rl_ij_split = split_to_components(rl_ij, self.lmax, dim=1)

            pairs = []
            for l in range(len(EQ_i_split)):
                if self.update_info["rej"]:
                    EQ_i_l = self.vector_rejection(EQ_i_split[l], rl_ij_split[l])
                    EK_j_l = self.vector_rejection(EK_j_split[l], -rl_ij_split[l])
                else:
                    EQ_i_l = EQ_i_split[l]
                    EK_j_l = EK_j_split[l]
                pairs.append((EQ_i_l, EK_j_l))
        elif not self.update_info["rej"]:
            pairs = [(EQ_i, EK_j)]
        else:
            EQr_i = self.vector_rejection(EQ_i, rl_ij)
            EKr_j = self.vector_rejection(EK_j, -rl_ij)
            pairs = [(EQr_i, EKr_j)]

        # Compute edge weights
        w_ij = None
        for el in pairs:
            EQ_i_l, EK_j_l = el
            w_l = (EQ_i_l * EK_j_l).sum(dim=1)
            if w_ij is None:
                w_ij = w_l
            else:
                w_ij = w_ij + w_l

        return self.gamma_t(t_ij) * self.gamma_w(w_ij)

    def aggregate(
        self,
        features: Tuple[Tensor, Tensor],
        index: Tensor,
        ptr: Optional[Tensor],
        dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor]:
        """
        Aggregate messages from source nodes to target nodes.
        
        This method implements the aggregation step of message passing, combining
        messages from neighboring nodes according to the specified aggregation method.
        
        Args:
            features: Tuple of scalar and vector features (h, X)
            index: Index tensor for scatter operation
            ptr: Pointer tensor for scatter operation
            dim_size: Dimension size for scatter operation
            
        Returns:
            Tuple containing:
                - Aggregated scalar features [num_nodes, 1, hidden_dims]
                - Aggregated high-degree steerable features [num_nodes, (L_max ** 2) - 1, hidden_dims]
        """
        h, X = features
        h = scatter(h, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        X = scatter(X, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return h, X

    def update(
        self,
        inputs: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        Update node features with aggregated messages.
        
        This method implements the update step of message passing. In this implementation,
        it simply passes through the aggregated features without additional processing.
        
        Args:
            inputs: Tuple of aggregated scalar and high-degree steerable features
            
        Returns:
            Tuple containing:
                - Updated scalar features [num_nodes, 1, hidden_dims]
                - Updated high-degree steerable features [num_nodes, (L_max ** 2) - 1, hidden_dims]
        """
        return inputs


class EQFF(nn.Module):
    """
    Equivariant Feed-Forward (EQFF) Network for mixing atom features.
    
    This module facilitates efficient channel-wise interaction while maintaining equivariance.
    It separates scalar and high-degree steerable features, allowing for specialized processing
    of each feature type before combining them with non-linear mappings as described in the paper:
    
    EQFF(h, X^(l)) = (h + m_1, X^(l) + m_2 * (X^(l)W_{vu}))
    where m_1, m_2 = split_2(gamma_{m}(||X^(l)W_{vu}||_2, h))
    """

    def __init__(
        self,
        n_atom_basis: int,
        activation: Callable,
        lmax: int,
        epsilon: float = 1e-8,
        weight_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_
    ):
        """
        Initialize EQFF module.
        
        Args:
            n_atom_basis: Number of features to describe atomic environments.
            activation: Activation function. If None, no activation function is used.
            lmax: Maximum angular momentum.
            epsilon: Stability constant added in norm to prevent numerical instabilities.
            weight_init: Weight initialization function.
            bias_init: Bias initialization function.
        """
        super(EQFF, self).__init__()
        self.lmax = lmax
        self.n_atom_basis = n_atom_basis
        self.epsilon = epsilon

        InitDense = partial(Dense, weight_init=weight_init, bias_init=bias_init)

        context_dim = 2 * n_atom_basis
        out_size = 2

        # gamma_m implementation
        self.gamma_m = nn.Sequential(
            InitDense(context_dim, n_atom_basis, activation=activation),
            InitDense(n_atom_basis, out_size * n_atom_basis, activation=None),
        )

        self.W_vu = InitDense(
            n_atom_basis, n_atom_basis, activation=None, bias=False
        )

    def reset_parameters(self):
        """Reset all learnable parameters of the module."""
        self.W_vu.reset_parameters()
        for l in self.gamma_m:
            l.reset_parameters()

    def forward(self, h: Tensor, X: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute intraatomic mixing.

        Args:
            h: Scalar input values, [num_nodes, 1, hidden_dims].
            X: High-degree steerable features, [num_nodes, (L_max ** 2) - 1, hidden_dims].

        Returns:
            Tuple of updated scalar values and high-degree steerable features,
            each of shape [num_nodes, 1, hidden_dims] and [num_nodes, (L_max ** 2) - 1, hidden_dims].
        """
        X_p = self.W_vu(X)

        # Compute norm of X_V with numerical stability
        X_pn = torch.sqrt(torch.sum(X_p**2, dim=-2, keepdim=True) + self.epsilon)

        # Concatenate features for context
        channel_context = [h, X_pn]
        ctx = torch.cat(channel_context, dim=-1)

        # Apply gamma_m transformation
        x = self.gamma_m(ctx)

        # Split output into scalar and vector components
        m1, m2 = torch.split(x, self.n_atom_basis, dim=-1)
        dX_intra = m2 * X_p

        # Update features with residual connections
        h = h + m1
        X = X + dX_intra

        return h, X



class GotenNet(nn.Module):
    """
    Graph Attention Transformer Network for atomic systems.
    
    GotenNet processes and updates two types of node features (invariant and steerable)
    and edge features (invariant) through three main mechanisms:
    
    1. GATA (Graph Attention Transformer Architecture): A degree-wise attention-based 
       message passing layer that updates both invariant and steerable features while 
       preserving equivariance.
    2. HTR (Hierarchical Tensor Refinement): Updates edge features across degrees with 
       inner products of steerable features.
    3. EQFF (Equivariant Feed-Forward): Further processes both types of node features
       while maintaining equivariance.
    """

    def __init__(
        self,
        n_atom_basis: int = 128,
        n_interactions: int = 8,
        radial_basis: Union[Callable, str] = 'expnorm',
        n_rbf: int = 32,
        cutoff_fn: Optional[Union[Callable, str]] = None,
        activation: Optional[Union[Callable, str]] = F.silu,
        max_z: int = 100,
        epsilon: float = 1e-8,
        weight_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        layernorm: str = "",
        steerable_norm: str = "",
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        edge_updates: Union[bool, str] = True,
        scale_edge: bool = True,
        lmax: int = 1,
        aggr: str = "add",
        evec_dim: Optional[int] = None,
        emlp_dim: Optional[int] = None,
        sep_htr: bool = True,
        sep_dir: bool = False,
        sep_tensor: bool = False,
    ):
        """
        Initialize GotenNet model.
        
        Args:
            n_atom_basis: Number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: Number of interaction blocks.
            radial_basis: Layer for expanding interatomic distances in a basis set.
            n_rbf: Number of radial basis functions.
            cutoff_fn: Cutoff function.
            activation: Activation function.
            max_z: Maximum atomic number.
            epsilon: Stability constant added in norm to prevent numerical instabilities.
            weight_init: Weight initialization function.
            bias_init: Bias initialization function.
            max_num_neighbors: Maximum number of neighbors.
            layernorm: Type of layer normalization to use.
            steerable_norm: Type of steerable normalization to use.
            num_heads: Number of attention heads.
            attn_dropout: Dropout probability for attention.
            edge_updates: Whether to update edge features.
            scale_edge: Whether to scale edge features.
            lmax: Maximum angular momentum.
            aggr: Aggregation method ('add', 'mean' or 'max').
            evec_dim: Dimension of edge vector features.
            emlp_dim: Dimension of edge MLP features.
            sep_htr: Whether to separate vector features in interaction.
            sep_dir: Whether to separate direction features.
            sep_tensor: Whether to separate tensor features.
        """
        super(GotenNet, self).__init__()

        self.scale_edge = scale_edge
        if type(weight_init) == str:
            weight_init = get_weight_init_by_string(weight_init)

        if type(bias_init) == str:
            bias_init = get_weight_init_by_string(bias_init)

        if type(activation) is str:
            activation = str2act(activation)

        self.n_atom_basis = self.hidden_dim = n_atom_basis
        self.n_interactions = n_interactions
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff


        self.node_init = NodeInit(
            [self.hidden_dim, self.hidden_dim], n_rbf, self.cutoff, max_z=max_z, weight_init=weight_init,
            bias_init=bias_init, proj_ln='layer', activation=activation
        )

        self.edge_init = EdgeInit(n_rbf, self.hidden_dim)

        radial_basis = str2basis(radial_basis)
        self.radial_basis = radial_basis(cutoff=self.cutoff, n_rbf=n_rbf)
        self.A_na = nn.Embedding(max_z, n_atom_basis, padding_idx=0)
        self.sphere = TensorInit(l=lmax)

        self.gata_list = nn.ModuleList([
            GATA(
                n_atom_basis=self.n_atom_basis, activation=activation, aggr=aggr,
                weight_init=weight_init, bias_init=bias_init,
                layer_norm=layernorm, steerable_norm=steerable_norm, cutoff=self.cutoff, epsilon=epsilon,
                num_heads=num_heads, dropout=attn_dropout,
                edge_updates=edge_updates, last_layer=(i == self.n_interactions - 1),
                scale_edge=scale_edge,
                evec_dim=evec_dim, emlp_dim=emlp_dim,
                sep_htr=sep_htr, sep_dir=sep_dir, sep_tensor=sep_tensor, lmax=lmax
            ) for i in range(self.n_interactions)
        ])

        self.eqff_list = nn.ModuleList([
            EQFF(
                n_atom_basis=self.n_atom_basis, activation=activation, lmax=lmax, epsilon=epsilon,
                weight_init=weight_init, bias_init=bias_init
            ) for i in range(self.n_interactions)
        ])

        self.reset_parameters()


    def reset_parameters(self):
        self.node_init.reset_parameters()
        self.edge_init.reset_parameters()
        for l in self.gata_list:
            l.reset_parameters()
        for l in self.eqff_list:
            l.reset_parameters()

    def forward(self, atomic_numbers, edge_index, edge_diff, edge_vec) -> Tuple[Tensor, Tensor]:
        """
        Compute atomic representations/embeddings.
        
        Args:
            atomic_numbers: Tensor of atomic numbers [num_nodes]
            edge_index: Tensor describing graph connectivity [2, num_edges]
            edge_diff: Tensor of edge distances [num_edges, 1]
            edge_vec: Tensor of edge direction vectors [num_edges, 3]
            
        Returns:
            Tuple containing:
                - Atomic representation [num_nodes, hidden_dims]
                - High-degree steerable features [num_nodes, (L_max ** 2) - 1, hidden_dims]
        """
        h = self.A_na(atomic_numbers)[:]
        phi_r0_ij = self.radial_basis(edge_diff)

        h = self.node_init(atomic_numbers, h, edge_index, edge_diff, phi_r0_ij)
        t_ij_init = self.edge_init(edge_index, phi_r0_ij, h)
        mask = edge_index[0] != edge_index[1]
        r0_ij = torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        edge_vec[mask] = edge_vec[mask] / r0_ij

        rl_ij = self.sphere(edge_vec)

        equi_dim = ((self.sphere.l + 1) ** 2) - 1
        # count number of edges for each node
        num_edges = scatter(torch.ones_like(edge_diff), edge_index[0], dim=0, reduce="sum")
        n_edges = num_edges[edge_index[0]]

        hs = h.shape
        X = torch.zeros((hs[0], equi_dim, hs[1]), device=h.device)
        h.unsqueeze_(1)
        t_ij = t_ij_init
        for _i, (gata, eqff) in enumerate(zip(self.gata_list, self.eqff_list, strict=False)):
            h, X, t_ij = gata(edge_index, h, X, rl_ij=rl_ij, t_ij=t_ij, r_ij=edge_diff, n_edges=n_edges) # idx_i, idx_j, n_atoms, # , f_ij=f_ij
            h, X = eqff(h, X)

        h = h.squeeze(1)
        return h, X


class GotenNetWrapper(GotenNet):
    """
    The wrapper around GotenNet for processing atomistic data. 
    """

    def __init__(
        self,
        *args, 
        max_num_neighbors=32,
        **kwargs
    ):
        super(GotenNetWrapper, self).__init__(*args, **kwargs)

        self.distance = Distance(self.cutoff, max_num_neighbors=max_num_neighbors, loop=True)
        self.reset_parameters()


    def forward(self, inputs: Mapping[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Compute atomic representations/embeddings.

        Args:
            inputs: Dictionary of input tensors containing atomic_numbers, pos, batch, 
                edge_index, r_ij, and dir_ij. Shape information:
                - atomic_numbers: [num_nodes]
                - pos: [num_nodes, 3]
                - batch: [num_nodes]
                - edge_index: [2, num_edges]

        Returns:
            Tuple containing:
                - Atomic representation [num_nodes, hidden_dims]
                - High-degree steerable features [num_nodes, (L_max ** 2) - 1, hidden_dims]
        """
        atomic_numbers, pos, batch = inputs.z, inputs.pos, inputs.batch
        edge_index, edge_diff, edge_vec = self.distance(pos, batch)
        return super().forward(atomic_numbers, edge_index, edge_diff, edge_vec)
