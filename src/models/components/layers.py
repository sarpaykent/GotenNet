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

    Args:
        lmax: The lmax value

    Returns:
        split_sizes: A list of split sizes for torch.split (sizes of spherical harmonic components)
    """

    return [2 * l + 1 for l in range(1, lmax + 1)]

class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

act_class_mapping = {"ssp": ShiftedSoftplus, "silu": nn.SiLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "swish": Swish}

def shifted_softplus(x: torch.Tensor):
    r"""Compute shifted soft-plus activation function.

    .. math::
       y = \ln\left(1 + e^{x}\right) - \ln(2)

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Shifted soft-plus of input.
    """
    return F.softplus(x) - math.log(2.0)

class PolynomialCutoff(nn.Module):
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
    return torch.sqrt(torch.sum(x ** 2, dim=dim, keepdim=keepdim)) + eps


class ScaleShift(nn.Module):
    r"""Scale and shift layer for standardization.

    .. math::
       y = x \times \sigma + \mu

    Args:
        mean (torch.Tensor): mean value :math:`\mu`.
        stddev (torch.Tensor): standard deviation value :math:`\sigma`.

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
    """Extraction layer to get an item from SchNetPack dictionary of input tensors.
    Args:
        key (str): Property to be extracted from SchNetPack input tensors.
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
    """Multiple layer fully connected perceptron neural network.
    Args:
        n_in (int): number of input nodes.
        n_out (int): number of output nodes.
        n_hidden (list of int or int, optional): number hidden layer nodes.
            If an integer, same number of node is used for all hidden layers resulting
            in a rectangular network.
            If None, the number of neurons is divided by two after each layer starting
            n_in resulting in a pyramidal network.
        n_layers (int, optional): number of layers.
        activation (callable, optional): activation function. All hidden layers would
            the same activation function except the output layer that does not apply
            any activation function.
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
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y


class GaussianRBF(nn.Module):
    r"""Gaussian radial basis functions."""

    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 0.0, trainable: bool = False
    ):
        r"""
        Args:
            n_rbf: total number of Gaussian functions, :math:`N_g`.
            cutoff: center of last Gaussian function, :math:`\mu_{N_g}`
            start: center of first Gaussian function, :math:`\mu_0`.
            trainable: If True, widths and offset of Gaussian functions
                are adjusted during training process.
        """
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
    Sine for radial basis expansion with coulomb decay. (0th order Bessel from DimeNet)
    """

    def __init__(self, cutoff=5.0, n_rbf=None, trainable=False):
        """
        Args:
            cutoff: radial cutoff
            n_rbf: number of basis functions.
        """
        super(BesselBasis, self).__init__()
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
    return glorot_orthogonal(tensor, scale=scale)


def _standardize(kernel):
    """
    Makes sure that Var(W) = 1 and E[W] = 0
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
    Generate a weight matrix with variance according to He initialization.
    Based on a random (semi-)orthogonal matrix neural networks
    are expected to learn better when features are decorrelated
    (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
    "Dropout: a simple way to prevent neural networks from overfitting",
    "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
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

    if init_str == '':
        # Noop
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
    r"""Fully connected linear layer with activation function.
    Borrowed from https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/nn/base.py

    .. math::
       y = \text{activation}(xW^T + b)

    Args:
        in_features (int): Number of input features :math:`x`.
        out_features (int): Number of output features :math:`y`.
        bias (bool, optional): If False, the layer will not adapt bias :math:`b`.
        activation (callable, optional): If None, no activation function is used.
        weight_init (callable, optional): Weight initializer from current weight.
        bias_init (callable, optional): Bias initializer from current bias.
        norm (str, optional): Normalization type ('layer', 'batch', 'instance', or None).
        gain (float, optional): Gain for weight initialization.
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
        hidden_dims (List[int]): List of hidden dimensions.
        bias (bool, optional): Whether to use bias.
        activation (callable, optional): Activation function for hidden layers.
        last_activation (callable, optional): Activation function for output layer.
        weight_init (callable, optional): Weight initialization function.
        bias_init (callable, optional): Bias initialization function.
        norm (str, optional): Normalization type.
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
    return s.lower().replace('-', '').replace('_', '').replace(' ', '')

# https://github.com/sunglasses-ai/classy/blob/3e74cba1fdf1b9f9f2ba1cfcfa6c2017aa59fc04/classy/optim/factories.py#L14
def get_activations(optional=False, *args, **kwargs):
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
    if selected not in options:
        raise ValueError(
            f'Invalid choice "{selected}", choose one from {", ".join(list(options.keys()))} '
        )

    activation = options[selected]
    if inspect.isclass(activation):
        activation =  activation()
    return activation

def str2act(input_str, *args, **kwargs):
    if input_str == "":
        return None

    act = get_activations(*args, optional=True,  **kwargs)
    out = dictionary_to_option(act, input_str)
    return out

class ExpNormalSmearing(nn.Module):
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
    if type(input_str) != str:
        return input_str

    if input_str == 'BesselBasis':
        radial_basis = BesselBasis
    elif input_str == 'GaussianRBF':
        radial_basis = GaussianRBF
    elif input_str.lower() == 'expnorm':
        radial_basis = ExpNormalSmearing
    else:
        raise ValueError('Unknown radial basis: {}'.format(input_str))

    return radial_basis





class TensorInit(nn.Module):

    def __init__(self, l=2):
        super(TensorInit, self).__init__()
        self.l = l

    def forward(self, edge_vec):
        edge_sh = self._calculate_components(self.l, edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2])
        return edge_sh

    @property
    def tensor_size(self):
        return ((self.l + 1) ** 2) - 1

    @staticmethod
    def _calculate_components(lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:

        sh_1_0, sh_1_1, sh_1_2 = x, y, z

        if lmax == 1:
            return torch.stack([sh_1_0, sh_1_1, sh_1_2], dim=-1)

        # (x^2, y^2, z^2) ^2

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

        if lmax == 2:
            return torch.stack([sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4], dim=-1)

        # Borrowed from e3nn: https://github.com/e3nn/e3nn/blob/main/e3nn/o3/_spherical_harmonics.py#L188
        sh_3_0 = (1 / 6) * math.sqrt(42) * (sh_2_0 * z + sh_2_4 * x)
        sh_3_1 = math.sqrt(7) * sh_2_0 * y
        sh_3_2 = (1 / 8) * math.sqrt(168) * (4.0 * y2 - x2z2) * x
        sh_3_3 = (1 / 2) * math.sqrt(7) * y * (2.0 * y2 - 3.0 * x2z2)
        sh_3_4 = (1 / 8) * math.sqrt(168) * z * (4.0 * y2 - x2z2)
        sh_3_5 = math.sqrt(7) * sh_2_4 * y
        sh_3_6 = (1 / 6) * math.sqrt(42) * (sh_2_4 * z - sh_2_0 * x)

        if lmax == 3:
            return torch.stack(
                [
                    sh_1_0,
                    sh_1_1,
                    sh_1_2,
                    sh_2_0,
                    sh_2_1,
                    sh_2_2,
                    sh_2_3,
                    sh_2_4,
                    sh_3_0,
                    sh_3_1,
                    sh_3_2,
                    sh_3_3,
                    sh_3_4,
                    sh_3_5,
                    sh_3_6,
                ],
                dim=-1,
            )

        sh_4_0 = (3 / 4) * math.sqrt(2) * (sh_3_0 * z + sh_3_6 * x)
        sh_4_1 = (3 / 4) * sh_3_0 * y + (3 / 8) * math.sqrt(6) * sh_3_1 * z + (3 / 8) * math.sqrt(6) * sh_3_5 * x
        sh_4_2 = (
                -3 / 56 * math.sqrt(14) * sh_3_0 * z
                + (3 / 14) * math.sqrt(21) * sh_3_1 * y
                + (3 / 56) * math.sqrt(210) * sh_3_2 * z
                + (3 / 56) * math.sqrt(210) * sh_3_4 * x
                + (3 / 56) * math.sqrt(14) * sh_3_6 * x
        )
        sh_4_3 = (
                -3 / 56 * math.sqrt(42) * sh_3_1 * z
                + (3 / 28) * math.sqrt(105) * sh_3_2 * y
                + (3 / 28) * math.sqrt(70) * sh_3_3 * x
                + (3 / 56) * math.sqrt(42) * sh_3_5 * x
        )
        sh_4_4 = -3 / 28 * math.sqrt(42) * sh_3_2 * x + (3 / 7) * math.sqrt(7) * sh_3_3 * y - 3 / 28 * math.sqrt(
            42) * sh_3_4 * z
        sh_4_5 = (
                -3 / 56 * math.sqrt(42) * sh_3_1 * x
                + (3 / 28) * math.sqrt(70) * sh_3_3 * z
                + (3 / 28) * math.sqrt(105) * sh_3_4 * y
                - 3 / 56 * math.sqrt(42) * sh_3_5 * z
        )
        sh_4_6 = (
                -3 / 56 * math.sqrt(14) * sh_3_0 * x
                - 3 / 56 * math.sqrt(210) * sh_3_2 * x
                + (3 / 56) * math.sqrt(210) * sh_3_4 * z
                + (3 / 14) * math.sqrt(21) * sh_3_5 * y
                - 3 / 56 * math.sqrt(14) * sh_3_6 * z
        )
        sh_4_7 = -3 / 8 * math.sqrt(6) * sh_3_1 * x + (3 / 8) * math.sqrt(6) * sh_3_5 * z + (3 / 4) * sh_3_6 * y
        sh_4_8 = (3 / 4) * math.sqrt(2) * (-sh_3_0 * x + sh_3_6 * z)
        if lmax == 4:
            return torch.stack(
                [
                    sh_1_0,
                    sh_1_1,
                    sh_1_2,
                    sh_2_0,
                    sh_2_1,
                    sh_2_2,
                    sh_2_3,
                    sh_2_4,
                    sh_3_0,
                    sh_3_1,
                    sh_3_2,
                    sh_3_3,
                    sh_3_4,
                    sh_3_5,
                    sh_3_6,
                    sh_4_0,
                    sh_4_1,
                    sh_4_2,
                    sh_4_3,
                    sh_4_4,
                    sh_4_5,
                    sh_4_6,
                    sh_4_7,
                    sh_4_8,
                ],
                dim=-1,
            )

        sh_5_0 = (1 / 10) * math.sqrt(110) * (sh_4_0 * z + sh_4_8 * x)
        sh_5_1 = (1 / 5) * math.sqrt(11) * sh_4_0 * y + (1 / 5) * math.sqrt(22) * sh_4_1 * z + (1 / 5) * math.sqrt(
            22) * sh_4_7 * x
        sh_5_2 = (
                -1 / 30 * math.sqrt(22) * sh_4_0 * z
                + (4 / 15) * math.sqrt(11) * sh_4_1 * y
                + (1 / 15) * math.sqrt(154) * sh_4_2 * z
                + (1 / 15) * math.sqrt(154) * sh_4_6 * x
                + (1 / 30) * math.sqrt(22) * sh_4_8 * x
        )
        sh_5_3 = (
                -1 / 30 * math.sqrt(66) * sh_4_1 * z
                + (1 / 15) * math.sqrt(231) * sh_4_2 * y
                + (1 / 30) * math.sqrt(462) * sh_4_3 * z
                + (1 / 30) * math.sqrt(462) * sh_4_5 * x
                + (1 / 30) * math.sqrt(66) * sh_4_7 * x
        )
        sh_5_4 = (
                -1 / 15 * math.sqrt(33) * sh_4_2 * z
                + (2 / 15) * math.sqrt(66) * sh_4_3 * y
                + (1 / 15) * math.sqrt(165) * sh_4_4 * x
                + (1 / 15) * math.sqrt(33) * sh_4_6 * x
        )
        sh_5_5 = (
                -1 / 15 * math.sqrt(110) * sh_4_3 * x + (1 / 3) * math.sqrt(11) * sh_4_4 * y - 1 / 15 * math.sqrt(
            110) * sh_4_5 * z
        )
        sh_5_6 = (
                -1 / 15 * math.sqrt(33) * sh_4_2 * x
                + (1 / 15) * math.sqrt(165) * sh_4_4 * z
                + (2 / 15) * math.sqrt(66) * sh_4_5 * y
                - 1 / 15 * math.sqrt(33) * sh_4_6 * z
        )
        sh_5_7 = (
                -1 / 30 * math.sqrt(66) * sh_4_1 * x
                - 1 / 30 * math.sqrt(462) * sh_4_3 * x
                + (1 / 30) * math.sqrt(462) * sh_4_5 * z
                + (1 / 15) * math.sqrt(231) * sh_4_6 * y
                - 1 / 30 * math.sqrt(66) * sh_4_7 * z
        )
        sh_5_8 = (
                -1 / 30 * math.sqrt(22) * sh_4_0 * x
                - 1 / 15 * math.sqrt(154) * sh_4_2 * x
                + (1 / 15) * math.sqrt(154) * sh_4_6 * z
                + (4 / 15) * math.sqrt(11) * sh_4_7 * y
                - 1 / 30 * math.sqrt(22) * sh_4_8 * z
        )
        sh_5_9 = -1 / 5 * math.sqrt(22) * sh_4_1 * x + (1 / 5) * math.sqrt(22) * sh_4_7 * z + (1 / 5) * math.sqrt(
            11) * sh_4_8 * y
        sh_5_10 = (1 / 10) * math.sqrt(110) * (-sh_4_0 * x + sh_4_8 * z)
        if lmax == 5:
            return torch.stack(
                [
                    sh_1_0,
                    sh_1_1,
                    sh_1_2,
                    sh_2_0,
                    sh_2_1,
                    sh_2_2,
                    sh_2_3,
                    sh_2_4,
                    sh_3_0,
                    sh_3_1,
                    sh_3_2,
                    sh_3_3,
                    sh_3_4,
                    sh_3_5,
                    sh_3_6,
                    sh_4_0,
                    sh_4_1,
                    sh_4_2,
                    sh_4_3,
                    sh_4_4,
                    sh_4_5,
                    sh_4_6,
                    sh_4_7,
                    sh_4_8,
                    sh_5_0,
                    sh_5_1,
                    sh_5_2,
                    sh_5_3,
                    sh_5_4,
                    sh_5_5,
                    sh_5_6,
                    sh_5_7,
                    sh_5_8,
                    sh_5_9,
                    sh_5_10,
                ],
                dim=-1,
            )

        sh_6_0 = (1 / 6) * math.sqrt(39) * (sh_5_0 * z + sh_5_10 * x)
        sh_6_1 = (
                (1 / 6) * math.sqrt(13) * sh_5_0 * y + (1 / 12) * math.sqrt(130) * sh_5_1 * z + (1 / 12) * math.sqrt(
            130) * sh_5_9 * x
        )
        sh_6_2 = (
                -1 / 132 * math.sqrt(286) * sh_5_0 * z
                + (1 / 33) * math.sqrt(715) * sh_5_1 * y
                + (1 / 132) * math.sqrt(286) * sh_5_10 * x
                + (1 / 44) * math.sqrt(1430) * sh_5_2 * z
                + (1 / 44) * math.sqrt(1430) * sh_5_8 * x
        )
        sh_6_3 = (
                -1 / 132 * math.sqrt(858) * sh_5_1 * z
                + (1 / 22) * math.sqrt(429) * sh_5_2 * y
                + (1 / 22) * math.sqrt(286) * sh_5_3 * z
                + (1 / 22) * math.sqrt(286) * sh_5_7 * x
                + (1 / 132) * math.sqrt(858) * sh_5_9 * x
        )
        sh_6_4 = (
                -1 / 66 * math.sqrt(429) * sh_5_2 * z
                + (2 / 33) * math.sqrt(286) * sh_5_3 * y
                + (1 / 66) * math.sqrt(2002) * sh_5_4 * z
                + (1 / 66) * math.sqrt(2002) * sh_5_6 * x
                + (1 / 66) * math.sqrt(429) * sh_5_8 * x
        )
        sh_6_5 = (
                -1 / 66 * math.sqrt(715) * sh_5_3 * z
                + (1 / 66) * math.sqrt(5005) * sh_5_4 * y
                + (1 / 66) * math.sqrt(3003) * sh_5_5 * x
                + (1 / 66) * math.sqrt(715) * sh_5_7 * x
        )
        sh_6_6 = (
                -1 / 66 * math.sqrt(2145) * sh_5_4 * x + (1 / 11) * math.sqrt(143) * sh_5_5 * y - 1 / 66 * math.sqrt(
            2145) * sh_5_6 * z
        )
        sh_6_7 = (
                -1 / 66 * math.sqrt(715) * sh_5_3 * x
                + (1 / 66) * math.sqrt(3003) * sh_5_5 * z
                + (1 / 66) * math.sqrt(5005) * sh_5_6 * y
                - 1 / 66 * math.sqrt(715) * sh_5_7 * z
        )
        sh_6_8 = (
                -1 / 66 * math.sqrt(429) * sh_5_2 * x
                - 1 / 66 * math.sqrt(2002) * sh_5_4 * x
                + (1 / 66) * math.sqrt(2002) * sh_5_6 * z
                + (2 / 33) * math.sqrt(286) * sh_5_7 * y
                - 1 / 66 * math.sqrt(429) * sh_5_8 * z
        )
        sh_6_9 = (
                -1 / 132 * math.sqrt(858) * sh_5_1 * x
                - 1 / 22 * math.sqrt(286) * sh_5_3 * x
                + (1 / 22) * math.sqrt(286) * sh_5_7 * z
                + (1 / 22) * math.sqrt(429) * sh_5_8 * y
                - 1 / 132 * math.sqrt(858) * sh_5_9 * z
        )
        sh_6_10 = (
                -1 / 132 * math.sqrt(286) * sh_5_0 * x
                - 1 / 132 * math.sqrt(286) * sh_5_10 * z
                - 1 / 44 * math.sqrt(1430) * sh_5_2 * x
                + (1 / 44) * math.sqrt(1430) * sh_5_8 * z
                + (1 / 33) * math.sqrt(715) * sh_5_9 * y
        )
        sh_6_11 = (
                -1 / 12 * math.sqrt(130) * sh_5_1 * x + (1 / 6) * math.sqrt(13) * sh_5_10 * y + (1 / 12) * math.sqrt(
            130) * sh_5_9 * z
        )
        sh_6_12 = (1 / 6) * math.sqrt(39) * (-sh_5_0 * x + sh_5_10 * z)
        if lmax == 6:
            return torch.stack(
                [
                    sh_1_0,
                    sh_1_1,
                    sh_1_2,
                    sh_2_0,
                    sh_2_1,
                    sh_2_2,
                    sh_2_3,
                    sh_2_4,
                    sh_3_0,
                    sh_3_1,
                    sh_3_2,
                    sh_3_3,
                    sh_3_4,
                    sh_3_5,
                    sh_3_6,
                    sh_4_0,
                    sh_4_1,
                    sh_4_2,
                    sh_4_3,
                    sh_4_4,
                    sh_4_5,
                    sh_4_6,
                    sh_4_7,
                    sh_4_8,
                    sh_5_0,
                    sh_5_1,
                    sh_5_2,
                    sh_5_3,
                    sh_5_4,
                    sh_5_5,
                    sh_5_6,
                    sh_5_7,
                    sh_5_8,
                    sh_5_9,
                    sh_5_10,
                    sh_6_0,
                    sh_6_1,
                    sh_6_2,
                    sh_6_3,
                    sh_6_4,
                    sh_6_5,
                    sh_6_6,
                    sh_6_7,
                    sh_6_8,
                    sh_6_9,
                    sh_6_10,
                    sh_6_11,
                    sh_6_12,
                ],
                dim=-1,
            )

        sh_7_0 = (1 / 14) * math.sqrt(210) * (sh_6_0 * z + sh_6_12 * x)
        sh_7_1 = (1 / 7) * math.sqrt(15) * sh_6_0 * y + (3 / 7) * math.sqrt(5) * sh_6_1 * z + (3 / 7) * math.sqrt(
            5) * sh_6_11 * x
        sh_7_2 = (
                -1 / 182 * math.sqrt(390) * sh_6_0 * z
                + (6 / 91) * math.sqrt(130) * sh_6_1 * y
                + (3 / 91) * math.sqrt(715) * sh_6_10 * x
                + (1 / 182) * math.sqrt(390) * sh_6_12 * x
                + (3 / 91) * math.sqrt(715) * sh_6_2 * z
        )
        sh_7_3 = (
                -3 / 182 * math.sqrt(130) * sh_6_1 * z
                + (3 / 182) * math.sqrt(130) * sh_6_11 * x
                + (3 / 91) * math.sqrt(715) * sh_6_2 * y
                + (5 / 182) * math.sqrt(858) * sh_6_3 * z
                + (5 / 182) * math.sqrt(858) * sh_6_9 * x
        )
        sh_7_4 = (
                (3 / 91) * math.sqrt(65) * sh_6_10 * x
                - 3 / 91 * math.sqrt(65) * sh_6_2 * z
                + (10 / 91) * math.sqrt(78) * sh_6_3 * y
                + (15 / 182) * math.sqrt(78) * sh_6_4 * z
                + (15 / 182) * math.sqrt(78) * sh_6_8 * x
        )
        sh_7_5 = (
                -5 / 91 * math.sqrt(39) * sh_6_3 * z
                + (15 / 91) * math.sqrt(39) * sh_6_4 * y
                + (3 / 91) * math.sqrt(390) * sh_6_5 * z
                + (3 / 91) * math.sqrt(390) * sh_6_7 * x
                + (5 / 91) * math.sqrt(39) * sh_6_9 * x
        )
        sh_7_6 = (
                -15 / 182 * math.sqrt(26) * sh_6_4 * z
                + (12 / 91) * math.sqrt(65) * sh_6_5 * y
                + (2 / 91) * math.sqrt(1365) * sh_6_6 * x
                + (15 / 182) * math.sqrt(26) * sh_6_8 * x
        )
        sh_7_7 = (
                -3 / 91 * math.sqrt(455) * sh_6_5 * x + (1 / 13) * math.sqrt(195) * sh_6_6 * y - 3 / 91 * math.sqrt(
            455) * sh_6_7 * z
        )
        sh_7_8 = (
                -15 / 182 * math.sqrt(26) * sh_6_4 * x
                + (2 / 91) * math.sqrt(1365) * sh_6_6 * z
                + (12 / 91) * math.sqrt(65) * sh_6_7 * y
                - 15 / 182 * math.sqrt(26) * sh_6_8 * z
        )
        sh_7_9 = (
                -5 / 91 * math.sqrt(39) * sh_6_3 * x
                - 3 / 91 * math.sqrt(390) * sh_6_5 * x
                + (3 / 91) * math.sqrt(390) * sh_6_7 * z
                + (15 / 91) * math.sqrt(39) * sh_6_8 * y
                - 5 / 91 * math.sqrt(39) * sh_6_9 * z
        )
        sh_7_10 = (
                -3 / 91 * math.sqrt(65) * sh_6_10 * z
                - 3 / 91 * math.sqrt(65) * sh_6_2 * x
                - 15 / 182 * math.sqrt(78) * sh_6_4 * x
                + (15 / 182) * math.sqrt(78) * sh_6_8 * z
                + (10 / 91) * math.sqrt(78) * sh_6_9 * y
        )
        sh_7_11 = (
                -3 / 182 * math.sqrt(130) * sh_6_1 * x
                + (3 / 91) * math.sqrt(715) * sh_6_10 * y
                - 3 / 182 * math.sqrt(130) * sh_6_11 * z
                - 5 / 182 * math.sqrt(858) * sh_6_3 * x
                + (5 / 182) * math.sqrt(858) * sh_6_9 * z
        )
        sh_7_12 = (
                -1 / 182 * math.sqrt(390) * sh_6_0 * x
                + (3 / 91) * math.sqrt(715) * sh_6_10 * z
                + (6 / 91) * math.sqrt(130) * sh_6_11 * y
                - 1 / 182 * math.sqrt(390) * sh_6_12 * z
                - 3 / 91 * math.sqrt(715) * sh_6_2 * x
        )
        sh_7_13 = -3 / 7 * math.sqrt(5) * sh_6_1 * x + (3 / 7) * math.sqrt(5) * sh_6_11 * z + (1 / 7) * math.sqrt(
            15) * sh_6_12 * y
        sh_7_14 = (1 / 14) * math.sqrt(210) * (-sh_6_0 * x + sh_6_12 * z)
        if lmax == 7:
            return torch.stack(
                [
                    sh_1_0,
                    sh_1_1,
                    sh_1_2,
                    sh_2_0,
                    sh_2_1,
                    sh_2_2,
                    sh_2_3,
                    sh_2_4,
                    sh_3_0,
                    sh_3_1,
                    sh_3_2,
                    sh_3_3,
                    sh_3_4,
                    sh_3_5,
                    sh_3_6,
                    sh_4_0,
                    sh_4_1,
                    sh_4_2,
                    sh_4_3,
                    sh_4_4,
                    sh_4_5,
                    sh_4_6,
                    sh_4_7,
                    sh_4_8,
                    sh_5_0,
                    sh_5_1,
                    sh_5_2,
                    sh_5_3,
                    sh_5_4,
                    sh_5_5,
                    sh_5_6,
                    sh_5_7,
                    sh_5_8,
                    sh_5_9,
                    sh_5_10,
                    sh_6_0,
                    sh_6_1,
                    sh_6_2,
                    sh_6_3,
                    sh_6_4,
                    sh_6_5,
                    sh_6_6,
                    sh_6_7,
                    sh_6_8,
                    sh_6_9,
                    sh_6_10,
                    sh_6_11,
                    sh_6_12,
                    sh_7_0,
                    sh_7_1,
                    sh_7_2,
                    sh_7_3,
                    sh_7_4,
                    sh_7_5,
                    sh_7_6,
                    sh_7_7,
                    sh_7_8,
                    sh_7_9,
                    sh_7_10,
                    sh_7_11,
                    sh_7_12,
                    sh_7_13,
                    sh_7_14,
                ],
                dim=-1,
            )

        sh_8_0 = (1 / 4) * math.sqrt(17) * (sh_7_0 * z + sh_7_14 * x)
        sh_8_1 = (
                (1 / 8) * math.sqrt(17) * sh_7_0 * y + (1 / 16) * math.sqrt(238) * sh_7_1 * z + (1 / 16) * math.sqrt(
            238) * sh_7_13 * x
        )
        sh_8_2 = (
                -1 / 240 * math.sqrt(510) * sh_7_0 * z
                + (1 / 60) * math.sqrt(1785) * sh_7_1 * y
                + (1 / 240) * math.sqrt(46410) * sh_7_12 * x
                + (1 / 240) * math.sqrt(510) * sh_7_14 * x
                + (1 / 240) * math.sqrt(46410) * sh_7_2 * z
        )
        sh_8_3 = (
                (1 / 80)
                * math.sqrt(2)
                * (
                        -math.sqrt(85) * sh_7_1 * z
                        + math.sqrt(2210) * sh_7_11 * x
                        + math.sqrt(85) * sh_7_13 * x
                        + math.sqrt(2210) * sh_7_2 * y
                        + math.sqrt(2210) * sh_7_3 * z
                )
        )
        sh_8_4 = (
                (1 / 40) * math.sqrt(935) * sh_7_10 * x
                + (1 / 40) * math.sqrt(85) * sh_7_12 * x
                - 1 / 40 * math.sqrt(85) * sh_7_2 * z
                + (1 / 10) * math.sqrt(85) * sh_7_3 * y
                + (1 / 40) * math.sqrt(935) * sh_7_4 * z
        )
        sh_8_5 = (
                (1 / 48)
                * math.sqrt(2)
                * (
                        math.sqrt(102) * sh_7_11 * x
                        - math.sqrt(102) * sh_7_3 * z
                        + math.sqrt(1122) * sh_7_4 * y
                        + math.sqrt(561) * sh_7_5 * z
                        + math.sqrt(561) * sh_7_9 * x
                )
        )
        sh_8_6 = (
                (1 / 16) * math.sqrt(34) * sh_7_10 * x
                - 1 / 16 * math.sqrt(34) * sh_7_4 * z
                + (1 / 4) * math.sqrt(17) * sh_7_5 * y
                + (1 / 16) * math.sqrt(102) * sh_7_6 * z
                + (1 / 16) * math.sqrt(102) * sh_7_8 * x
        )
        sh_8_7 = (
                -1 / 80 * math.sqrt(1190) * sh_7_5 * z
                + (1 / 40) * math.sqrt(1785) * sh_7_6 * y
                + (1 / 20) * math.sqrt(255) * sh_7_7 * x
                + (1 / 80) * math.sqrt(1190) * sh_7_9 * x
        )
        sh_8_8 = (
                -1 / 60 * math.sqrt(1785) * sh_7_6 * x + (1 / 15) * math.sqrt(255) * sh_7_7 * y - 1 / 60 * math.sqrt(
            1785) * sh_7_8 * z
        )
        sh_8_9 = (
                -1 / 80 * math.sqrt(1190) * sh_7_5 * x
                + (1 / 20) * math.sqrt(255) * sh_7_7 * z
                + (1 / 40) * math.sqrt(1785) * sh_7_8 * y
                - 1 / 80 * math.sqrt(1190) * sh_7_9 * z
        )
        sh_8_10 = (
                -1 / 16 * math.sqrt(34) * sh_7_10 * z
                - 1 / 16 * math.sqrt(34) * sh_7_4 * x
                - 1 / 16 * math.sqrt(102) * sh_7_6 * x
                + (1 / 16) * math.sqrt(102) * sh_7_8 * z
                + (1 / 4) * math.sqrt(17) * sh_7_9 * y
        )
        sh_8_11 = (
                (1 / 48)
                * math.sqrt(2)
                * (
                        math.sqrt(1122) * sh_7_10 * y
                        - math.sqrt(102) * sh_7_11 * z
                        - math.sqrt(102) * sh_7_3 * x
                        - math.sqrt(561) * sh_7_5 * x
                        + math.sqrt(561) * sh_7_9 * z
                )
        )
        sh_8_12 = (
                (1 / 40) * math.sqrt(935) * sh_7_10 * z
                + (1 / 10) * math.sqrt(85) * sh_7_11 * y
                - 1 / 40 * math.sqrt(85) * sh_7_12 * z
                - 1 / 40 * math.sqrt(85) * sh_7_2 * x
                - 1 / 40 * math.sqrt(935) * sh_7_4 * x
        )
        sh_8_13 = (
                (1 / 80)
                * math.sqrt(2)
                * (
                        -math.sqrt(85) * sh_7_1 * x
                        + math.sqrt(2210) * sh_7_11 * z
                        + math.sqrt(2210) * sh_7_12 * y
                        - math.sqrt(85) * sh_7_13 * z
                        - math.sqrt(2210) * sh_7_3 * x
                )
        )
        sh_8_14 = (
                -1 / 240 * math.sqrt(510) * sh_7_0 * x
                + (1 / 240) * math.sqrt(46410) * sh_7_12 * z
                + (1 / 60) * math.sqrt(1785) * sh_7_13 * y
                - 1 / 240 * math.sqrt(510) * sh_7_14 * z
                - 1 / 240 * math.sqrt(46410) * sh_7_2 * x
        )
        sh_8_15 = (
                -1 / 16 * math.sqrt(238) * sh_7_1 * x + (1 / 16) * math.sqrt(238) * sh_7_13 * z + (1 / 8) * math.sqrt(
            17) * sh_7_14 * y
        )
        sh_8_16 = (1 / 4) * math.sqrt(17) * (-sh_7_0 * x + sh_7_14 * z)
        if lmax == 8:
            return torch.stack(
                [
                    sh_1_0,
                    sh_1_1,
                    sh_1_2,
                    sh_2_0,
                    sh_2_1,
                    sh_2_2,
                    sh_2_3,
                    sh_2_4,
                    sh_3_0,
                    sh_3_1,
                    sh_3_2,
                    sh_3_3,
                    sh_3_4,
                    sh_3_5,
                    sh_3_6,
                    sh_4_0,
                    sh_4_1,
                    sh_4_2,
                    sh_4_3,
                    sh_4_4,
                    sh_4_5,
                    sh_4_6,
                    sh_4_7,
                    sh_4_8,
                    sh_5_0,
                    sh_5_1,
                    sh_5_2,
                    sh_5_3,
                    sh_5_4,
                    sh_5_5,
                    sh_5_6,
                    sh_5_7,
                    sh_5_8,
                    sh_5_9,
                    sh_5_10,
                    sh_6_0,
                    sh_6_1,
                    sh_6_2,
                    sh_6_3,
                    sh_6_4,
                    sh_6_5,
                    sh_6_6,
                    sh_6_7,
                    sh_6_8,
                    sh_6_9,
                    sh_6_10,
                    sh_6_11,
                    sh_6_12,
                    sh_7_0,
                    sh_7_1,
                    sh_7_2,
                    sh_7_3,
                    sh_7_4,
                    sh_7_5,
                    sh_7_6,
                    sh_7_7,
                    sh_7_8,
                    sh_7_9,
                    sh_7_10,
                    sh_7_11,
                    sh_7_12,
                    sh_7_13,
                    sh_7_14,
                    sh_8_0,
                    sh_8_1,
                    sh_8_2,
                    sh_8_3,
                    sh_8_4,
                    sh_8_5,
                    sh_8_6,
                    sh_8_7,
                    sh_8_8,
                    sh_8_9,
                    sh_8_10,
                    sh_8_11,
                    sh_8_12,
                    sh_8_13,
                    sh_8_14,
                    sh_8_15,
                    sh_8_16,
                ],
                dim=-1,
            )




class TensorLayerNorm(nn.Module):
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
    def __init__(self, cutoff, max_num_neighbors=32, loop=True, direction="source_to_target"):
        super(Distance, self).__init__()
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
    
    Initializes node features based on atom types and their local environment.
    
    Args:
        hidden_channels: Dimension of hidden channels.
        num_rbf: Number of radial basis functions.
        cutoff: Cutoff distance for interactions.
        max_z: Maximum atomic number.
        activation: Activation function.
        proj_ln: Projection layer normalization.
        weight_init: Weight initialization function.
        bias_init: Bias initialization function.
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
    
    Initializes edge features based on connected nodes and radial basis functions.
    
    Args:
        num_rbf: Number of radial basis functions.
        hidden_channels: Dimension of hidden channels.
        activation: Activation function.
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
