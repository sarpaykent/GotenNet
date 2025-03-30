from typing import Optional, Union

import ase
import torch
import torch.nn.functional as F
import torch_scatter
from torch import nn
from torch.autograd import grad
from torch_geometric.utils import scatter

from gotennet.models.components.layers import (
    Dense,
    GetItem,
    ScaleShift,
    SchnetMLP,
    shifted_softplus,
    str2act,
)
from gotennet.utils import get_logger

log = get_logger(__name__)


class GatedEquivariantBlock(nn.Module):
    """
    The gated equivariant block is used to obtain rotationally invariant and equivariant features to be used
    for tensorial prop.
    """

    def __init__(
        self,
        n_sin: int,
        n_vin: int,
        n_sout: int,
        n_vout: int,
        n_hidden: int,
        activation=F.silu,
        sactivation=None,
    ):
        """
        Initialize the GatedEquivariantBlock.
        
        Args:
            n_sin (int): Input dimension of scalar features.
            n_vin (int): Input dimension of vectorial features.
            n_sout (int): Output dimension of scalar features.
            n_vout (int): Output dimension of vectorial features.
            n_hidden (int): Size of hidden layers.
            activation: Activation of hidden layers.
            sactivation: Final activation to scalar features.
        """
        super().__init__()
        self.n_sin = n_sin
        self.n_vin = n_vin
        self.n_sout = n_sout
        self.n_vout = n_vout
        self.n_hidden = n_hidden
        self.mix_vectors = Dense(n_vin, 2 * n_vout, activation=None, bias=False)
        self.scalar_net = nn.Sequential(
            Dense(
                n_sin + n_vout, n_hidden, activation=activation
            ),
            Dense(n_hidden, n_sout + n_vout, activation=None),
        )
        self.sactivation = sactivation

    def forward(self, scalars: torch.Tensor, vectors: torch.Tensor):
        """
        Forward pass of the GatedEquivariantBlock.
        
        Args:
            scalars (torch.Tensor): Scalar input features.
            vectors (torch.Tensor): Vector input features.
            
        Returns:
            tuple: Tuple containing:
                - torch.Tensor: Output scalar features.
                - torch.Tensor: Output vector features.
        """
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



class AtomwiseV3(nn.Module):
    """
    Atomwise prediction module V3 for predicting atomic properties.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        aggregation_mode: Optional[str] = "sum",
        n_layers: int = 2,
        n_hidden: Optional[int] = None,
        activation = shifted_softplus,
        property: str = "y",
        contributions: Optional[str] = None,
        derivative: Optional[str] = None,
        negative_dr: bool = True,
        create_graph: bool = True,
        mean: Optional[Union[float, torch.Tensor]] = None,
        stddev: Optional[Union[float, torch.Tensor]] = None,
        atomref: Optional[torch.Tensor] = None,
        outnet: Optional[nn.Module] = None,
        return_vector: Optional[str] = None,
        standardize: bool = True,
    ):
        """
        Initialize the AtomwiseV3 module.
        
        Args:
            n_in (int): Input dimension of atomwise features.
            n_out (int): Output dimension of target property.
            aggregation_mode (Optional[str]): Aggregation method for atomic contributions.
            n_layers (int): Number of layers in the output network.
            n_hidden (Optional[int]): Size of hidden layers.
            activation: Activation function.
            property (str): Name of the target property.
            contributions (Optional[str]): Name of the atomic contributions.
            derivative (Optional[str]): Name of the property derivative.
            negative_dr (bool): If True, negative derivative of the energy.
            create_graph (bool): If True, create computational graph for derivatives.
            mean (Optional[Union[float, torch.Tensor]]): Mean of the property for standardization.
            stddev (Optional[Union[float, torch.Tensor]]): Standard deviation for standardization.
            atomref (Optional[torch.Tensor]): Reference single-atom properties.
            outnet (Optional[nn.Module]): Network for property prediction.
            return_vector (Optional[str]): Name of the vector property to return.
            standardize (bool): If True, standardize the output property.
        """
        super(AtomwiseV3, self).__init__()

        self.return_vector = return_vector
        self.n_layers = n_layers
        self.create_graph = create_graph
        self.property = property
        self.contributions = contributions
        self.derivative = derivative
        self.negative_dr = negative_dr
        self.standardize = standardize


        mean = 0.0 if mean is None else mean
        stddev = 1.0 if stddev is None else stddev
        self.mean = mean
        self.stddev = stddev

        if type(activation) is str:
            activation = str2act(activation)

        if atomref is not None:
            self.atomref = nn.Embedding.from_pretrained(
                atomref.type(torch.float32)
            )
        else:
            self.atomref = None

        if outnet is None:
            self.out_net = nn.Sequential(
                GetItem("representation"),
                SchnetMLP(n_in, n_out, n_hidden, n_layers, activation),
            )
        else:
            self.out_net = outnet

        # build standardization layer
        if self.standardize and (mean is not None and stddev is not None):
            self.standardize = ScaleShift(mean, stddev)
        else:
            self.standardize = nn.Identity()

        self.aggregation_mode = aggregation_mode

    def forward(self, inputs):
        """
        Predicts atomwise property.
        
        Args:
            inputs: Input data containing atomic representations.
            
        Returns:
            dict: Dictionary with predicted properties.
        """
        atomic_numbers = inputs.z
        result = {}
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
            result[self.derivative] = dy
        return result


class Atomwise(nn.Module):
    """
    Atomwise prediction module for predicting atomic properties.
    """
    
    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        aggregation_mode: Optional[str] = "sum",
        n_layers: int = 2,
        n_hidden: Optional[int] = None,
        activation = shifted_softplus,
        property: str = "y",
        contributions: Optional[str] = None,
        derivative: Optional[str] = None,
        negative_dr: bool = True,
        create_graph: bool = True,
        mean: Optional[torch.Tensor] = None,
        stddev: Optional[torch.Tensor] = None,
        atomref: Optional[torch.Tensor] = None,
        outnet: Optional[nn.Module] = None,
        return_vector: Optional[str] = None,
        standardize: bool = True,
    ):
        """
        Initialize the Atomwise module.
        
        Args:
            n_in (int): Input dimension of atomwise features.
            n_out (int): Output dimension of target property.
            aggregation_mode (Optional[str]): Aggregation method for atomic contributions.
            n_layers (int): Number of layers in the output network.
            n_hidden (Optional[int]): Size of hidden layers.
            activation: Activation function.
            property (str): Name of the target property.
            contributions (Optional[str]): Name of the atomic contributions.
            derivative (Optional[str]): Name of the property derivative.
            negative_dr (bool): If True, negative derivative of the energy.
            create_graph (bool): If True, create computational graph for derivatives.
            mean (Optional[torch.Tensor]): Mean of the property for standardization.
            stddev (Optional[torch.Tensor]): Standard deviation for standardization.
            atomref (Optional[torch.Tensor]): Reference single-atom properties.
            outnet (Optional[nn.Module]): Network for property prediction.
            return_vector (Optional[str]): Name of the vector property to return.
            standardize (bool): If True, standardize the output property.
        """
        super(Atomwise, self).__init__()

        self.return_vector = return_vector
        self.n_layers = n_layers
        self.create_graph = create_graph
        self.property = property
        self.contributions = contributions
        self.derivative = derivative
        self.negative_dr = negative_dr
        self.standardize = standardize
        
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
        """
        Predicts atomwise property.
        
        Args:
            inputs: Input data containing atomic representations.
            
        Returns:
            dict: Dictionary with predicted properties.
        """
        atomic_numbers = inputs.z
        result = {}
        
        if self.equivariant:
            l0 = inputs.representation
            l1 = inputs.vector_representation
            for eqlayer in self.out_net:
                l0, l1 = eqlayer(l0, l1)

            if self.return_vector:
                result[self.return_vector] = l1
            yi = l0
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


class Dipole(nn.Module):
    """Output layer for dipole moment."""

    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[int] = None,
        activation = F.silu,
        property: str = "dipole",
        predict_magnitude: bool = False,
        output_v: bool = True,
        mean: Optional[torch.Tensor] = None,
        stddev: Optional[torch.Tensor] = None,
    ):
        """
        Initialize the Dipole module.
        
        Args:
            n_in (int): Input dimension of atomwise features.
            n_hidden (Optional[int]): Size of hidden layers.
            activation: Activation function.
            property (str): Name of property to be predicted.
            predict_magnitude (bool): If true, calculate magnitude of dipole.
            output_v (bool): If true, output vector representation.
            mean (Optional[torch.Tensor]): Mean of the property for standardization.
            stddev (Optional[torch.Tensor]): Standard deviation for standardization.
        """
        super().__init__()

        self.stddev = stddev
        self.mean = mean
        self.output_v = output_v
        if n_hidden is None:
            n_hidden = n_in

        self.property = property
        self.derivative = None
        self.predict_magnitude = predict_magnitude

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
        """
        Predicts dipole moment.
        
        Args:
            inputs: Input data containing atomic representations.
            
        Returns:
            dict: Dictionary with predicted dipole properties.
        """
        positions = inputs.pos
        l0 = inputs.representation
        l1 = inputs.vector_representation[:, :3, :]


        for eqlayer in self.equivariant_layers:
            l0, l1 = eqlayer(l0, l1)

        if self.stddev is not None:
            l0 =  self.stddev * l0 + self.mean

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
    """Electronic spatial extent prediction module."""
    
    def __init__(
        self,
        n_in: int,
        n_layers: int = 2,
        n_hidden: Optional[int] = None,
        activation = shifted_softplus,
        property: str = "y",
        contributions: Optional[str] = None,
        mean: Optional[torch.Tensor] = None,
        stddev: Optional[torch.Tensor] = None,
        outnet: Optional[nn.Module] = None,
    ):
        """
        Initialize the ElectronicSpatialExtentV2 module.
        
        Args:
            n_in (int): Input dimension of atomwise features.
            n_layers (int): Number of layers in the output network.
            n_hidden (Optional[int]): Size of hidden layers.
            activation: Activation function.
            property (str): Name of the target property.
            contributions (Optional[str]): Name of the atomic contributions.
            mean (Optional[torch.Tensor]): Mean of the property for standardization.
            stddev (Optional[torch.Tensor]): Standard deviation for standardization.
            outnet (Optional[nn.Module]): Network for property prediction.
        """
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
        )
        atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
        self.register_buffer("atomic_mass", atomic_mass)

    def forward(self, inputs):
        """
        Predicts the electronic spatial extent.
        
        Args:
            inputs: Input data containing atomic representations and positions.
            
        Returns:
            dict: Dictionary with predicted electronic spatial extent properties.
        """
        positions = inputs.pos
        x = self.out_net(inputs)
        mass = self.atomic_mass[inputs.z].view(-1, 1)
        c = scatter(mass * positions, inputs.batch, dim=0) / scatter(mass, inputs.batch, dim=0)

        yi = torch.norm(positions - c[inputs.batch], dim=1, keepdim=True)
        yi = yi ** 2 * x

        y = torch_scatter.scatter(yi, inputs.batch, dim=0, reduce=self.aggregation_mode)

        # collect results
        result = {self.property: y}

        if self.contributions:
            result[self.contributions] = x

        return result
