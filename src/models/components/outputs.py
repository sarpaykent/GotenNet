from typing import Optional, Union, Callable

import ase
import torch
import torch.nn.functional as F
import torch_scatter
from torch import nn
from torch.autograd import grad
from torch.nn.init import xavier_uniform_, zeros_
from torch_geometric.utils import scatter

from src.models.components.ops import (Dense, ScaleShift, GetItem, SchnetMLP, shifted_softplus, str2act,
                                       centralize, decentralize)
from src.utils import (
    RankedLogger,
)

log = RankedLogger(__name__, rank_zero_only=True)


class SNNDense(nn.Linear):
    """Fully connected linear layer with activation function."""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            activation: Union[Callable, nn.Module] = None,
            weight_init: Callable = xavier_uniform_,
            bias_init: Callable = zeros_,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(SNNDense, self).__init__(in_features, out_features, bias)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def reset_parameters(self):
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input: torch.Tensor):
        y = F.linear(input, self.weight, self.bias)
        y = self.activation(y)
        return y


class GatedEquivariantBlock(nn.Module):
    """
    The gated equivariant block is used to obtain rotationally invariant and equivariant features
    for tensorial properties.
    """

    def __init__(self, n_sin, n_vin, n_sout, n_vout, n_hidden, activation=F.silu, sactivation=None):
        super().__init__()
        self.n_sin = n_sin
        self.n_vin = n_vin
        self.n_sout = n_sout
        self.n_vout = n_vout
        self.n_hidden = n_hidden
        self.mix_vectors = SNNDense(n_vin, 2 * n_vout, activation=None, bias=False)
        self.scalar_net = nn.Sequential(
            Dense(
                n_sin + n_vout, n_hidden, activation=activation
            ),
            Dense(n_hidden, n_sout + n_vout, activation=None),
        )
        self.sactivation = sactivation

    def forward(self, scalars, vectors):
        vmix = self.mix_vectors(vectors)
        vectors_V, vectors_W = torch.split(vmix, self.n_vout, dim=-1)
        vectors_Vn = torch.norm(vectors_V, dim=-2)

        ctx = torch.cat([scalars, vectors_Vn], dim=-1)
        x = self.scalar_net(ctx)
        s_out, x = torch.split(x, [self.n_sout, self.n_vout], dim=-1)
        v_out = x.unsqueeze(-2) * vectors_W

        if self.sactivation:
            s_out = self.sactivation(s_out)

        return s_out, v_out


class PassThrough(nn.Module):
    """Pass through layer that returns the input as output."""

    def __init__(
            self,
            n_in=None,
            n_hidden=None,
            property="y",
            derivative=None,
            negative_dr=True,
            create_graph=True,
    ):
        super(PassThrough, self).__init__()
        self.property = property
        self.derivative = derivative
        self.negative_dr = negative_dr
        self.create_graph = create_graph

        pass

    def forward(self, inputs):
        result = {}
        # inputs.representation
        result[self.property] = inputs.representation
        if self.derivative:
            sign = -1.0 if self.negative_dr else 1.0
            dy = grad(
                outputs=result[self.property],
                inputs=[inputs.pos],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=self.create_graph,
                retain_graph=True
            )[0]

            dy = sign * dy

            result[self.derivative] = dy

        return result


class AtomwiseV3(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.
    """

    def __init__(
            self,
            n_in,
            n_out=1,
            aggregation_mode: Optional[str] = "sum",
            n_layers=2,
            n_hidden=None,
            activation=shifted_softplus,
            property="y",
            contributions=None,
            derivative=None,
            negative_dr=True,
            create_graph=True,
            mean=None,
            stddev=None,
            atomref=None,
            outnet=None,
            return_vector=None,
            standardize=True,
            equiscalar=False,
            vec_cont=0.0,
    ):
        super(AtomwiseV3, self).__init__()

        self.vec_cont = vec_cont
        self.return_vector = return_vector
        self.n_layers = n_layers
        self.create_graph = create_graph
        self.property = property
        self.contributions = contributions
        self.derivative = derivative
        self.negative_dr = negative_dr
        self.standardize = standardize
        self.equiscalar = equiscalar

        mean = 0.0 if mean is None else mean
        stddev = 1.0 if stddev is None else stddev
        self.mean = mean
        self.stddev = stddev

        if type(activation) is str:
            activation = str2act(activation)

        # initialize single atom energies
        if atomref is not None:
            self.atomref = nn.Embedding.from_pretrained(
                atomref.type(torch.float32)
            )
        else:
            self.atomref = None

        self.equivariant = False
        # build output network
        if outnet is None:
            self.out_net = nn.Sequential(
                GetItem("representation"),
                SchnetMLP(n_in, n_out, n_hidden, n_layers, activation),
            )
        elif outnet == 'equi':
            if n_hidden is None:
                n_hidden = n_in
            self.out_net = nn.ModuleList(
                [
                    GatedEquivariantBlock(n_sin=n_in, n_vin=n_in, n_sout=n_hidden, n_vout=n_hidden, n_hidden=n_hidden,
                                          activation=activation,
                                          sactivation=activation),
                    GatedEquivariantBlock(n_sin=n_hidden, n_vin=n_hidden, n_sout=1, n_vout=1,
                                          n_hidden=n_hidden, activation=activation)
                ])
            self.equivariant = True
        else:
            self.out_net = outnet

        # build standardization layer
        if self.standardize and (mean is not None and stddev is not None):
            self.standardize = ScaleShift(mean, stddev)
        else:
            self.standardize = nn.Identity()

        self.aggregation_mode = aggregation_mode

    def forward(self, inputs):
        r"""
        predicts atomwise property
        """
        atomic_numbers = inputs.z
        result = {}
        # run prediction
        if self.equivariant:
            l0 = inputs.representation
            l1 = inputs.vector_representation
            for eqlayer in self.out_net:
                # print(l0.shape, l1.shape, '1')
                l0, l1 = eqlayer(l0, l1)

            if self.return_vector:
                result[self.return_vector] = l1
            yi = l0
            if self.equiscalar:
                yi = l0 + l1.sum() * 0
        else:
            yi = self.out_net(inputs)
        yi = yi * self.stddev

        if self.atomref is not None:
            y0 = self.atomref(atomic_numbers)
            yi = yi + y0

        if self.aggregation_mode is not None:
            y = torch_scatter.scatter(yi, inputs.batch, dim=0, reduce=self.aggregation_mode)
        else:
            y = yi

        y = y + self.mean

        # collect results
        result[self.property] = y

        if self.contributions:
            result[self.contributions] = yi
        if self.derivative:
            sign = -1.0 if self.negative_dr else 1.0
            dy = grad(
                outputs=result[self.property],
                inputs=[inputs.pos],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=self.create_graph,
                retain_graph=True
            )[0]

            dy = sign * dy
            if self.vec_cont > 0.0:
                dy = dy + self.vec_cont * l1.squeeze(-1)

            result[self.derivative] = dy
        return result


class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.
    """

    def __init__(
            self,
            n_in,
            n_out=1,
            aggregation_mode: Optional[str] = "sum",
            n_layers=2,
            n_hidden=None,
            activation=shifted_softplus,
            property="y",
            contributions=None,
            derivative=None,
            negative_dr=True,
            create_graph=True,
            mean=None,
            stddev=None,
            atomref=None,
            outnet=None,
            return_vector=None,
            standardize=True,
            equiscalar=False,
            eq_vec_only=False
    ):
        super(Atomwise, self).__init__()

        self.eq_vec_only = eq_vec_only
        self.return_vector = return_vector
        self.n_layers = n_layers
        self.create_graph = create_graph
        self.property = property
        self.contributions = contributions
        self.derivative = derivative
        self.negative_dr = negative_dr
        self.standardize = standardize
        self.equiscalar = equiscalar

        mean = torch.FloatTensor([0.0]) if mean is None else mean
        stddev = torch.FloatTensor([1.0]) if stddev is None else stddev

        if type(activation) is str:
            activation = str2act(activation)

        # initialize single atom energies
        if atomref is not None:
            self.atomref = nn.Embedding.from_pretrained(
                atomref.type(torch.float32)
            )
        else:
            self.atomref = None

        self.equivariant = False
        # build output network
        if outnet is None:
            self.out_net = nn.Sequential(
                GetItem("representation"),
                SchnetMLP(n_in, n_out, n_hidden, n_layers, activation),
            )
        elif outnet == 'equi':
            if n_hidden is None:
                n_hidden = n_in
            self.out_net = nn.ModuleList(
                [
                    GatedEquivariantBlock(n_sin=n_in, n_vin=n_in, n_sout=n_hidden, n_vout=n_hidden, n_hidden=n_hidden,
                                          activation=activation,
                                          sactivation=activation),
                    GatedEquivariantBlock(n_sin=n_hidden, n_vin=n_hidden, n_sout=1, n_vout=1,
                                          n_hidden=n_hidden, activation=activation)
                ])
            self.equivariant = True
        else:
            self.out_net = outnet

        # build standardization layer
        if self.standardize and (mean is not None and stddev is not None):
            log.info(f"Using standardization with mean {mean} and stddev {stddev}")
            self.standardize = ScaleShift(mean, stddev)
        else:
            self.standardize = nn.Identity()

        self.aggregation_mode = aggregation_mode

    def forward(self, inputs):
        atomic_numbers = inputs.z
        result = {}
        # run prediction
        if self.equivariant:
            l0 = inputs.representation
            l1 = inputs.vector_representation
            for eqlayer in self.out_net:
                l0, l1 = eqlayer(l0, l1)

            if self.return_vector:
                result[self.return_vector] = l1
            yi = l0
            if self.equiscalar:
                yi = l0 + l1.sum() * 0
        else:
            yi = self.out_net(inputs)
        yi = self.standardize(yi)

        if self.atomref is not None:
            y0 = self.atomref(atomic_numbers)
            yi = yi + y0

        if self.aggregation_mode is not None:
            y = torch_scatter.scatter(yi, inputs.batch, dim=0, reduce=self.aggregation_mode)
        else:
            y = yi

        # collect results
        result[self.property] = y

        if self.contributions:
            result[self.contributions] = yi

        if self.derivative:
            sign = -1.0 if self.negative_dr else 1.0
            dy = grad(
                outputs=result[self.property],
                inputs=[inputs.pos],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=self.create_graph,
                retain_graph=True
            )[0]

            result[self.derivative] = sign * dy
        return result


#


class Dipole(nn.Module):
    """ Output layer for dipole moment """

    def __init__(
            self,
            n_in,
            n_hidden,
            activation=F.silu,
            property="dipole",
            predict_magnitude=False,
            decentralize_pos=False,
            output_v=True,
            mean=None,
            stddev=None,
            centralize=False,
    ):
        super().__init__()

        self.stddev = stddev
        self.mean = mean
        self.centralize = centralize
        self.output_v = output_v
        if n_hidden is None:
            n_hidden = n_in

        self.property = property
        self.derivative = None
        self.predict_magnitude = predict_magnitude
        self.decentralize_pos = decentralize_pos

        self.equivariant_layers = nn.ModuleList(
            [
                GatedEquivariantBlock(n_sin=n_in, n_vin=n_in, n_sout=n_hidden, n_vout=n_hidden, n_hidden=n_hidden,
                                      activation=activation,
                                      sactivation=activation),
                GatedEquivariantBlock(n_sin=n_hidden, n_vin=n_hidden, n_sout=1, n_vout=1,
                                      n_hidden=n_hidden, activation=activation)
            ])
        self.requires_dr = False
        self.requires_stress = False
        self.aggregation_mode = 'sum'

    def forward(self, inputs):
        if self.centralize:
            centroid, positions = centralize(inputs, 'pos', inputs.batch)
        else:
            positions = inputs.pos

        l0 = inputs.representation
        l1 = inputs.vector_representation[:, :3, :]

        if self.decentralize_pos:
            if 'centroid' in inputs:
                positions = decentralize(positions, inputs.batch, inputs.centroid)
            else:
                raise ValueError("decentralize is set to True, but no centroid is given in inputs.")

        for eqlayer in self.equivariant_layers:
            l0, l1 = eqlayer(l0, l1)

        if self.stddev is not None:
            l0 = self.stddev * l0 + self.mean

        atomic_dipoles = torch.squeeze(l1, -1)
        charges = l0
        dipole_offsets = positions * charges

        y = atomic_dipoles + dipole_offsets
        # y = torch.sum(y, dim=1)
        y = torch_scatter.scatter(y, inputs.batch, dim=0, reduce=self.aggregation_mode)
        if self.output_v:
            y_vector = torch_scatter.scatter(l1, inputs.batch, dim=0, reduce=self.aggregation_mode)

        if self.predict_magnitude:
            y = torch.norm(y, dim=1, keepdim=True)

        result = {self.property: y}
        if self.output_v:
            result[self.property + "_vector"] = y_vector
        return result


class ElectronicSpatialExtentV2(Atomwise):
    """
    Predicts the electronic spatial extent using a formalism close to the dipole moment layer.
    """

    def __init__(
            self,
            n_in,
            n_layers=2,
            n_hidden=None,
            activation=shifted_softplus,
            property="y",
            contributions=None,
            mean=None,
            stddev=None,
            outnet=None,
            eq_vec_only=False
    ):
        super(ElectronicSpatialExtentV2, self).__init__(
            n_in,
            1,
            "sum",
            n_layers,
            n_hidden,
            activation=activation,
            mean=mean,
            stddev=stddev,
            outnet=outnet,
            property=property,
            contributions=contributions,
            eq_vec_only=eq_vec_only
        )
        atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
        self.register_buffer("atomic_mass", atomic_mass)

    def forward(self, inputs):
        """
        Predicts the electronic spatial extent.
        """
        positions = inputs.pos
        if self.equivariant:
            l0 = inputs.representation
            l1 = inputs.vector_representation

            if self.eq_vec_only:
                l1, _ = torch.split(l1, [3, 5], dim=1)
            for eqlayer in self.out_net:
                l0, l1 = eqlayer(l0, l1)
            x = l0
        else:
            x = self.out_net(inputs)
        mass = self.atomic_mass[inputs.z].view(-1, 1)
        c = scatter(mass * positions, inputs.batch, dim=0) / scatter(mass, inputs.batch, dim=0)

        yi = torch.norm(positions - c[inputs.batch], dim=1, keepdim=True)
        yi = yi ** 2 * x

        if self.equivariant:
            yi = yi + torch.norm(l1.squeeze(-1))

        y = torch_scatter.scatter(yi, inputs.batch, dim=0, reduce=self.aggregation_mode)

        # collect results
        result = {self.property: y}

        if self.contributions:
            result[self.contributions] = x

        return result
