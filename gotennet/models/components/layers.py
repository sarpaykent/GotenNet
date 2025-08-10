# Standard library imports
from __future__ import absolute_import, division, print_function

import inspect
import math
from functools import partial
from typing import List

# Related third-party imports
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn as nn
from torch.nn.init import constant_, xavier_uniform_
from torch_cluster import radius_graph
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot_orthogonal

# Local application/library specific imports
import gotennet.utils as utils

zeros_initializer = partial(constant_, val=0.0)
log = utils.get_logger(__name__)


def get_split_sizes_from_lmax(lmax):
    """
    Return split sizes for torch.split based on lmax.

    Calculates the dimensions of spherical harmonic components for each
    angular momentum value from 1 to lmax.

    Args:
        lmax: Maximum angular momentum value.

    Returns:
        List[int]: List of split sizes for torch.split.
    """
    return [2 * l + 1 for l in range(1, lmax + 1)]

class ShiftedSoftplus(nn.Module):
    """
    Shifted Softplus activation function.

    Computes `softplus(x) - log(2)`.
    """
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class Swish(nn.Module):
    """
    Swish activation function.

    Computes `x * sigmoid(x)`. Also known as SiLU.
    """
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

act_class_mapping = {"ssp": ShiftedSoftplus, "silu": nn.SiLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "swish": Swish}

def shifted_softplus(x: torch.Tensor):
    """
    Compute shifted soft-plus activation function.

    Computes `ln(1 + exp(x)) - ln(2)`.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Shifted soft-plus of input.
    """
    return F.softplus(x) - math.log(2.0)

class PolynomialCutoff(nn.Module):
    """
    Polynomial cutoff function, as proposed in DimeNet.

    Smoothly reduces values to zero based on a cutoff radius using a polynomial decay.
    Reference: https://arxiv.org/abs/2003.03123

    Args:
        cutoff (float): Cutoff radius.
        p (int, optional): Exponent for the polynomial decay. Defaults to 6.
    """
    def __init__(self, cutoff, p: int = 6):
        super(PolynomialCutoff, self).__init__()
        self.cutoff = cutoff
        self.p = p

    @staticmethod
    def polynomial_cutoff(
        r: Tensor,
        rcut: float,
        p: float = 6.0
    ) -> Tensor:
        """
        Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123
        """
        if not p >= 2.0:
            # replace below with logger error
            print(f"Exponent p={p} has to be >= 2.")
            print("Exiting code.")


            log.error(f"Exponent p={p} has to be >= 2.")
            log.error("Exiting code.")
            exit()

        rscaled = r / rcut

        out = 1.0
        out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(rscaled, p))
        out = out + (p * (p + 2.0) * torch.pow(rscaled, p + 1.0))
        out = out - ((p * (p + 1.0) / 2) * torch.pow(rscaled, p + 2.0))

        return out * (rscaled < 1.0).float()

    def forward(self, r):
        return self.polynomial_cutoff(r=r, rcut=self.cutoff, p=self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(cutoff={self.cutoff}, p={self.p})"

class CosineCutoff(nn.Module):
    """
    Cosine cutoff function.

    Smoothly reduces values to zero based on a cutoff radius using a cosine function.

    Args:
        cutoff (float): Cutoff radius.
    """
    def __init__(self, cutoff):
        super(CosineCutoff, self).__init__()

        if isinstance(cutoff, torch.Tensor):
            cutoff = cutoff.item()
        self.cutoff = cutoff

    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs


@torch.jit.script
def safe_norm(x: Tensor, dim:int=-2, eps:float=1e-8, keepdim: bool=False):
    """
    Compute the norm of a tensor safely, avoiding division by zero.

    Args:
        x (Tensor): Input tensor.
        dim (int, optional): Dimension along which to compute the norm. Defaults to -2.
        eps (float, optional): Small epsilon value for numerical stability. Defaults to 1e-8.
        keepdim (bool, optional): Whether the output tensor has `dim` retained or not. Defaults to False.

    Returns:
        Tensor: The norm of the input tensor.
    """
    return torch.sqrt(torch.sum(x ** 2, dim=dim, keepdim=keepdim)) + eps


class ScaleShift(nn.Module):
    """
    Scale and shift layer for standardization.

    Applies `y = x * stddev + mean`. Useful for normalizing outputs.

    Args:
        mean (torch.Tensor or float): Mean value (`mu`).
        stddev (torch.Tensor or float): Standard deviation value (`sigma`).
    """
    def __init__(self, mean, stddev):
        super(ScaleShift, self).__init__()
        if isinstance(mean, float):
            mean = torch.FloatTensor([mean])
        if isinstance(stddev, float):
            stddev = torch.FloatTensor([stddev])
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)

    def forward(self, input):
        """Compute layer output.

        Args:
            input (torch.Tensor): input data.

        Returns:
            torch.Tensor: layer output.

        """
        y = input * self.stddev + self.mean
        return y


class GetItem(nn.Module):
    """
    Extraction layer to get an item from a dictionary-like input.

    Args:
        key (str): Key of the item to be extracted from the input dictionary.
    """
    def __init__(self, key):
        super(GetItem, self).__init__()
        self.key = key

    def forward(self, inputs):
        """Compute layer output.
        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.
        Returns:
            torch.Tensor: layer output.
        """
        return inputs[self.key]

class SchnetMLP(nn.Module):
    """
    Multiple layer fully connected perceptron neural network, based on SchNet.

    Args:
        n_in (int): Number of input features.
        n_out (int): Number of output features.
        n_hidden (list of int or int, optional): Number of hidden layer nodes.
            If an integer, uses the same number for all hidden layers.
            If None, creates a pyramidal network where layer size is halved. Defaults to None.
        n_layers (int, optional): Total number of layers (including input and output). Defaults to 2.
        activation (callable, optional): Activation function for hidden layers. Defaults to shifted_softplus.
    """
    def __init__(
            self, n_in, n_out, n_hidden=None, n_layers=2, activation=shifted_softplus
    ):
        super(SchnetMLP, self).__init__()
        # get list of number of nodes in input, hidden & output layers
        if n_hidden is None:
            c_neurons = n_in
            self.n_neurons = []
            for _i in range(n_layers):
                self.n_neurons.append(c_neurons)
                c_neurons = c_neurons // 2
            self.n_neurons.append(n_out)
        else:
            # get list of number of nodes hidden layers
            if type(n_hidden) is int:
                n_hidden = [n_hidden] * (n_layers - 1)
            self.n_neurons = [n_in] + n_hidden + [n_out]

        # assign a Dense layer (with activation function) to each hidden layer
        layers = [
            Dense(self.n_neurons[i], self.n_neurons[i + 1], activation=activation)
            for i in range(n_layers - 1)
        ]
        # assign a Dense layer (without activation function) to the output layer
        layers.append(Dense(self.n_neurons[-2], self.n_neurons[-1], activation=None))
        # put all layers together to make the network
        self.out_net = nn.Sequential(*layers)

    def forward(self, inputs):
        """Compute neural network output.
        Args:
            inputs (torch.Tensor): network input.
        Returns:
            torch.Tensor: network output.
        """
        return self.out_net(inputs)


def gaussian_rbf(inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor):
    """
    Compute Gaussian radial basis functions.

    Args:
        inputs (torch.Tensor): Input distances. Shape: [..., 1]
        offsets (torch.Tensor): Centers of the Gaussian functions. Shape: [n_rbf]
        widths (torch.Tensor): Widths of the Gaussian functions. Shape: [n_rbf]

    Returns:
        torch.Tensor: Gaussian RBF values. Shape: [..., n_rbf]
    """
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y


class GaussianRBF(nn.Module):
    """
    Gaussian radial basis functions module.

    Expands distances using Gaussian functions centered at different offsets.

    Args:
        n_rbf (int): Total number of Gaussian functions.
        cutoff (float): Center of the last Gaussian function (maximum distance).
        start (float, optional): Center of the first Gaussian function. Defaults to 0.0.
        trainable (bool, optional): If True, widths and offsets are learnable parameters. Defaults to False.
    """
    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 0.0, trainable: bool = False
    ):
        super(GaussianRBF, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf)
        widths = torch.FloatTensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        )
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor):
        return gaussian_rbf(inputs, self.offsets, self.widths)


class BesselBasis(nn.Module):
    """
    Sine for radial basis expansion with coulomb decay (0th order Bessel functions).

    Used in DimeNet. Reference: https://arxiv.org/abs/2003.03123

    Args:
        cutoff (float, optional): Radial cutoff distance. Defaults to 5.0.
        n_rbf (int, optional): Number of basis functions. Defaults to None.
        trainable (bool, optional): Kept for compatibility, but parameters are not learnable. Defaults to False.
    """
    def __init__(self, cutoff=5.0, n_rbf=None, trainable=False):
        super(BesselBasis, self).__init__()
        if n_rbf is None:
            raise ValueError("n_rbf must be specified for BesselBasis")
        self.n_rbf = n_rbf
        # compute offset and width of Gaussian functions
        freqs = torch.arange(1, n_rbf + 1) * math.pi / cutoff
        self.register_buffer("freqs", freqs)
        self.register_buffer('norm1', torch.tensor(1.0))

    def forward(self, inputs):
        a = self.freqs[None,  :]
        inputs = inputs[..., None]
        ax = inputs * a
        sinax = torch.sin(ax)

        norm = torch.where(inputs == 0, self.norm1, inputs)
        y = sinax / norm

        return y




def glorot_orthogonal_wrapper_(tensor, scale=2.0):
    """
    Wrapper for glorot_orthogonal initialization.

    Args:
        tensor (Tensor): Tensor to initialize.
        scale (float, optional): Scaling factor. Defaults to 2.0.

    Returns:
        Tensor: Initialized tensor.
    """
    return glorot_orthogonal(tensor, scale=scale)


def _standardize(kernel):
    """
    Standardize a kernel tensor to have zero mean and unit variance.

    Ensures Var(W) = 1 and E[W] = 0.

    Args:
        kernel (Tensor): The kernel tensor to standardize.

    Returns:
        Tensor: The standardized kernel tensor.
    """
    eps = 1e-6

    if len(kernel.shape) == 3:
        axis = [0, 1]  # last dimension is output dimension
    else:
        axis = 1

    var, mean = torch.var_mean(kernel, dim=axis, unbiased=True, keepdim=True)
    kernel = (kernel - mean) / (var + eps) ** 0.5
    return kernel


def he_orthogonal_init(tensor):
    """
    Initialize weights using He initialization with an orthogonal basis.

    Combines He initialization variance scaling with an orthogonal matrix,
    aiming for better decorrelation of features.

    Args:
        tensor (Tensor): The weight tensor to initialize.

    Returns:
        Tensor: The initialized weight tensor.
    """
    tensor = torch.nn.init.orthogonal_(tensor)

    if len(tensor.shape) == 3:
        fan_in = tensor.shape[:-1].numel()
    else:
        fan_in = tensor.shape[1]

    with torch.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data *= (1 / fan_in) ** 0.5

    return tensor

def get_weight_init_by_string(init_str):
    """
    Get a weight initialization function based on its string name.

    Args:
        init_str (str): Name of the initialization method (e.g., 'zeros', 'xavier_uniform').

    Returns:
        Callable: The corresponding weight initialization function.

    Raises:
        ValueError: If the initialization string is unknown.
    """
    if init_str == '':
        # No-op
        return lambda x: x
    elif init_str  == 'zeros':
        return torch.nn.init.zeros_
    elif init_str == 'xavier_uniform':
        return torch.nn.init.xavier_uniform_
    elif init_str == 'glo_orthogonal':
        return glorot_orthogonal_wrapper_
    elif init_str == 'he_orthogonal':
        return he_orthogonal_init
    else:
        raise ValueError(f'Unknown initialization {init_str}')


# train.py -m label=mu,alpha,homo,lumo,r2,zpve,U0,U,H,G,Cv name='${label_str}_int6_glo-ort_3090' hydra.sweeper.n_jobs=1 model.representation.n_interactions=6 model.representation.weight_init=glo_orthogonal

class Dense(nn.Linear):
    """
    Fully connected linear layer with optional activation and normalization.
    Applies a linear transformation followed by optional normalization and activation.
    Borrowed from https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/nn/base.py

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If False, the layer will not adapt bias. Defaults to True.
        activation (callable, optional): Activation function. If None, no activation is used. Defaults to None.
        weight_init (callable, optional): Weight initializer. Defaults to xavier_uniform_.
        bias_init (callable, optional): Bias initializer. Defaults to zeros_initializer.
        norm (str, optional): Normalization type ('layer', 'batch', 'instance', or None). Defaults to None.
        gain (float, optional): Gain for weight initialization if applicable. Defaults to None.
    """
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
        norm=None,
        gain=None,
    ):
        # initialize linear layer y = xW^T + b
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.gain = gain
        super(Dense, self).__init__(in_features, out_features, bias)
        # Initialize activation function
        if inspect.isclass(activation):
            self.activation = activation()
        self.activation = activation

        if norm == 'layer':
            self.norm = nn.LayerNorm(out_features)
        elif norm == 'batch':
            self.norm = nn.BatchNorm1d(out_features)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm1d(out_features)
        else:
            self.norm = None

    def reset_parameters(self):
        """Reinitialize model weight and bias values."""
        if self.gain:
            self.weight_init(self.weight, gain=self.gain)
        else:
            self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        """Compute layer output.

        Args:
            inputs (dict of torch.Tensor): batch of input values.

        Returns:
            torch.Tensor: layer output.

        """
        # compute linear layer y = xW^T + b
        y = super(Dense, self).forward(inputs)
        if self.norm is not None:
            y = self.norm(y)
        # add activation function
        if self.activation:
            y = self.activation(y)
        return y



class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable hidden dimensions and activations.

    Args:
        hidden_dims (List[int]): List defining the dimensions of each layer,
            including input and output (e.g., [in_dim, hid1_dim, ..., out_dim]).
        bias (bool, optional): Whether to use bias in linear layers. Defaults to True.
        activation (callable, optional): Activation function for hidden layers. Defaults to None.
        last_activation (callable, optional): Activation function for the output layer. Defaults to None.
        weight_init (callable, optional): Weight initialization function. Defaults to xavier_uniform_.
        bias_init (callable, optional): Bias initialization function. Defaults to zeros_initializer.
        norm (str, optional): Normalization type ('layer', 'batch', 'instance', or ''). Defaults to ''.
    """
    def __init__(
        self,
        hidden_dims: List[int],
        bias=True,
        activation=None,
        last_activation=None,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
        norm='',
    ):
        super().__init__()

        # hidden_dims = [hidden, half, hidden]

        dims = hidden_dims
        n_layers = len(dims)

        DenseMLP = partial(Dense, bias=bias, weight_init=weight_init, bias_init=bias_init)

        self.dense_layers = nn.ModuleList([
                DenseMLP(dims[i], dims[i + 1], activation=activation, norm=norm)
                for i in range(n_layers - 2)
            ] + [DenseMLP(dims[-2], dims[-1], activation=last_activation)])

        self.layers = nn.Sequential(*self.dense_layers)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.dense_layers:
            m.reset_parameters()


    def forward(self, x):
        return self.layers(x)


def normalize_string(s: str) -> str:
    """
    Normalize a string by converting to lowercase and removing dashes, underscores, and spaces.

    Args:
        s (str): Input string.

    Returns:
        str: Normalized string.
    """
    return s.lower().replace('-', '').replace('_', '').replace(' ', '')

def get_activations(optional=False, *args, **kwargs):
    """
    Get a dictionary mapping normalized activation function names to their classes/functions.

    Includes common activations from torch.nn and custom ones like shifted_softplus.
    Reference: https://github.com/sunglasses-ai/classy/blob/3e74cba1fdf1b9f9f2ba1cfcfa6c2017aa59fc04/classy/optim/factories.py#L14

    Args:
        optional (bool, optional): If True, include an empty string key mapping to None. Defaults to False.

    Returns:
        Dict[str, Optional[Callable]]: Dictionary mapping names to activation functions/classes.
    """
    activations = {
        normalize_string(act.__name__): act for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, torch.nn.Module)
    }
    activations.update({
        "relu": torch.nn.ReLU,
        "elu": torch.nn.ELU,
        "sigmoid": torch.nn.Sigmoid,
        "silu": torch.nn.SiLU,
        "mish": torch.nn.Mish,
        "swish": torch.nn.SiLU,
        "selu": torch.nn.SELU,
        "softplus": shifted_softplus,
    })


    if optional:
        activations[""] = None

    return activations


def get_activations_none(optional=False, *args, **kwargs):
    """
    Get a dictionary mapping normalized activation function names to their classes/functions,
    excluding softplus-based activations.

    Args:
        optional (bool, optional): If True, include an empty string and None key mapping to None. Defaults to False.

    Returns:
        Dict[str, Optional[Callable]]: Dictionary mapping names to activation functions/classes.
    """
    activations = {
        normalize_string(act.__name__): act for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, torch.nn.Module)
    }
    activations.update({
        "relu": torch.nn.ReLU,
        "elu": torch.nn.ELU,
        "sigmoid": torch.nn.Sigmoid,
        "silu": torch.nn.SiLU,
        "selu": torch.nn.SELU,
    })

    if optional:
        activations[""] = None
        activations[None] = None

    return activations


def dictionary_to_option(options, selected):
    """
    Select an option from a dictionary based on a key, handling potential class instantiation.

    Args:
        options (Dict): Dictionary of options (e.g., activation functions).
        selected (Optional[str]): The key of the selected option.

    Returns:
        Optional[Callable]: The selected option (possibly instantiated if it's a class).

    Raises:
        ValueError: If the selected key is not in the options dictionary.
    """
    if selected not in options:
        raise ValueError(
            f'Invalid choice "{selected}", choose one from {", ".join(list(options.keys()))}'
        )

    activation = options[selected]
    if inspect.isclass(activation):
        activation =  activation()
    return activation

def str2act(input_str, *args, **kwargs):
    """
    Convert an activation function name string to the corresponding function/class instance.

    Args:
        input_str (Optional[str]): Name of the activation function (case-insensitive, ignores '-', '_', ' ').
                                   If None or "", returns None.

    Returns:
        Optional[Callable]: The instantiated activation function or None.
    """
    if not input_str: # Handles None and ""
        return None

    act = get_activations(*args, optional=True,  **kwargs)
    out = dictionary_to_option(act, input_str)
    return out

class ExpNormalSmearing(nn.Module):
    """
    Exponential Normal Smearing for radial basis functions.

    Uses exponentially spaced means and Gaussian functions for smearing distances.

    Args:
        cutoff (float, optional): Cutoff distance. Defaults to 5.0.
        n_rbf (int, optional): Number of radial basis functions. Defaults to 50.
        trainable (bool, optional): If True, means and betas are learnable parameters. Defaults to False.
    """
    def __init__(self, cutoff=5.0, n_rbf=50, trainable=False):
        super(ExpNormalSmearing, self).__init__()
        if isinstance(cutoff, torch.Tensor):
            cutoff = cutoff.item()
        self.cutoff = cutoff
        self.n_rbf = n_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.n_rbf)
        betas = torch.tensor([(2 / self.n_rbf * (1 - start_value)) ** -2] * self.n_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(-self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2)


def str2basis(input_str):
    """
    Convert a radial basis function name string to the corresponding class.

    Args:
        input_str (Union[str, Callable]): Name of the basis function ('BesselBasis', 'GaussianRBF', 'expnorm')
                                          or already a callable class.

    Returns:
        Callable: The radial basis function class.

    Raises:
        ValueError: If the input string is unknown.
    """
    if not isinstance(input_str, str):
        return input_str # Assume it's already a callable class

    normalized_input = normalize_string(input_str)

    if normalized_input == 'besselbasis':
        radial_basis = BesselBasis
    elif input_str == 'GaussianRBF':
        radial_basis = GaussianRBF
    elif input_str.lower() == 'expnorm':
        radial_basis = ExpNormalSmearing
    else:
        raise ValueError('Unknown radial basis: {}'.format(input_str))

    return radial_basis

class TensorLayerNorm(nn.Module):
    """
    Layer normalization for high-degree steerable features (tensors).

    Applies normalization independently to each degree component of the tensor features.
    Uses max-min normalization within each degree component.

    Args:
        hidden_channels (int): Dimension of the feature channels.
        trainable (bool): Whether the scaling weight is learnable.
        lmax (int, optional): Maximum degree (lmax) of the tensor features. Defaults to 1.
    """
    def __init__(self, hidden_channels, trainable, lmax=1, **kwargs):
        super(TensorLayerNorm, self).__init__()

        self.hidden_channels = hidden_channels
        self.eps = 1e-12
        self.lmax = lmax

        weight = torch.ones(self.hidden_channels)
        if trainable:
            self.register_parameter("weight", nn.Parameter(weight))
        else:
            self.register_buffer("weight", weight)

        self.reset_parameters()

    def reset_parameters(self):
        weight = torch.ones(self.hidden_channels)
        self.weight.data.copy_(weight)


    def max_min_norm(self, tensor):
        # Based on VisNet (https://www.nature.com/articles/s41467-023-43720-2)
        dist = torch.norm(tensor, dim=1, keepdim=True)

        if (dist == 0).all():
            return torch.zeros_like(tensor)

        dist = dist.clamp(min=self.eps)
        direct = tensor / dist

        max_val, _ = torch.max(dist, dim=-1)
        min_val, _ = torch.min(dist, dim=-1)
        delta = (max_val - min_val).view(-1)
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        dist = (dist - min_val.view(-1, 1, 1)) / delta.view(-1, 1, 1)

        return F.relu(dist) * direct

    def forward(self, tensor):
        try:
            split_sizes = get_split_sizes_from_lmax(self.lmax)
        except ValueError as e:
            raise ValueError(f"TensorLayerNorm received unsupported feature dimension {tensor.shape[1]}: {str(e)}") from e

        # Split the vector into parts
        vec_parts = torch.split(tensor, split_sizes, dim=1)

        # Normalize each part separately
        normalized_parts = [self.max_min_norm(part) for part in vec_parts]

        # Concatenate the normalized parts
        normalized_vec = torch.cat(normalized_parts, dim=1)

        # Apply weight
        return normalized_vec * self.weight.unsqueeze(0).unsqueeze(0)


class Distance(nn.Module):
    """
    Compute edge distances and vectors between nodes within a cutoff radius.

    Uses torch_cluster.radius_graph to find neighbors.

    Args:
        cutoff (float): Cutoff distance for finding neighbors.
        max_num_neighbors (int, optional): Maximum number of neighbors to consider for each node. Defaults to 32.
        loop (bool, optional): Whether to include self-loops in the graph. Defaults to True.
        direction (str, optional): Direction of edge vectors ('source_to_target' or 'target_to_source').
                                   Defaults to "source_to_target".
    """
    def __init__(self, cutoff, max_num_neighbors=32, loop=True, direction="source_to_target"):
        super(Distance, self).__init__()
        if direction not in ["source_to_target", "target_to_source"]:
            raise ValueError(f"Unknown direction '{direction}'. Choose 'source_to_target' or 'target_to_source'.")
        self.direction = direction
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.loop = loop

    def forward(self, pos, batch):
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, loop=self.loop,
                                  max_num_neighbors=self.max_num_neighbors)
        if self.direction == "source_to_target":
            # keep as is
            edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        else:
            edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        if self.loop:
            mask = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
            edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)

        return edge_index, edge_weight, edge_vec


class NodeInit(MessagePassing):
    """
    Node initialization layer for message passing networks.

    Initializes scalar node features based on atom types and their local environment
    using message passing. Implements Eq. 1 and 2 from the GotenNet paper.

    Args:
        hidden_channels (Union[int, List[int]]): Dimension of hidden channels. If a list, defines MLP layers.
        num_rbf (int): Number of radial basis functions used for edge features.
        cutoff (float): Cutoff distance for interactions.
        max_z (int, optional): Maximum atomic number for embedding lookup. Defaults to 100.
        activation (Callable, optional): Activation function. Defaults to F.silu.
        proj_ln (str, optional): Type of layer normalization for projection ('layer' or ''). Defaults to ''.
        weight_init (Callable, optional): Weight initialization function. Defaults to nn.init.xavier_uniform_.
        bias_init (Callable, optional): Bias initialization function. Defaults to nn.init.zeros_.
    """
    def __init__(
        self,
        hidden_channels,
        num_rbf,
        cutoff,
        max_z=100,
        activation=F.silu,
        proj_ln='',
        weight_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_
    ):
        super(NodeInit, self).__init__(aggr="add")
        if type(hidden_channels) == int:
            hidden_channels = [hidden_channels]

        last_channel = hidden_channels[-1]
        self.A_nbr = nn.Embedding(max_z, last_channel)
        self.W_ndp = MLP(
            [num_rbf] + [last_channel], activation=None, norm='', weight_init=weight_init,
            bias_init=bias_init, last_activation=None
        )

        self.W_nrd_nru = MLP(
            [2*last_channel] + hidden_channels, activation=activation, norm=proj_ln,
            weight_init=weight_init, bias_init=bias_init, last_activation=None
        )
        self.cutoff = CosineCutoff(cutoff)
        self.reset_parameters()

    def reset_parameters(self):
        self.A_nbr.reset_parameters()
        self.W_ndp.reset_parameters()
        self.W_nrd_nru.reset_parameters()

    def forward(self, z, h, edge_index, r0_ij, varphi_r0_ij):
        # remove self loops
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            r0_ij = r0_ij[mask]
            varphi_r0_ij = varphi_r0_ij[mask]

        h_src = self.A_nbr(z)
        phi_r0_ij = self.cutoff(r0_ij)
        r0_ij_feat = self.W_ndp(varphi_r0_ij) * phi_r0_ij.view(-1, 1)

        # propagate_type: (h_src: Tensor, r0_ij_feat:Tensor)
        m_i = self.propagate(edge_index, h_src=h_src, r0_ij_feat=r0_ij_feat, size=None)
        return self.W_nrd_nru(torch.cat([h, m_i], dim=1))

    def message(self, h_src_j, r0_ij_feat):
        return h_src_j * r0_ij_feat

class EdgeInit(MessagePassing):
    """
    Edge initialization layer for message passing networks.

    Initializes scalar edge features based on connected node features and radial basis functions.
    Implements Eq. 3 from the GotenNet paper.

    Args:
        num_rbf (int): Number of radial basis functions.
        hidden_channels (int): Dimension of hidden channels (must match node features).
        activation (Callable, optional): Activation function (currently unused). Defaults to None.
    """
    def __init__(
        self,
        num_rbf,
        hidden_channels,
        activation=None
    ):
        super(EdgeInit, self).__init__(aggr=None)
        self.W_erp = nn.Linear(num_rbf, hidden_channels)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_erp.weight)
        self.W_erp.bias.data.fill_(0)

    def forward(self, edge_index, phi_r0_ij, h):
        # propagate_type: (h: Tensor, phi_r0_ij: Tensor)
        out = self.propagate(edge_index, h=h, phi_r0_ij=phi_r0_ij)
        return out

    def message(self, h_i, h_j, phi_r0_ij):
        return (h_i + h_j) * self.W_erp(phi_r0_ij)

    def aggregate(self, features, index):
        # no aggregate
        return features
