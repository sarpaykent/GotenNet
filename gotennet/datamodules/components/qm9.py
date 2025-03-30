import torch
from torch_geometric.datasets import QM9 as QM9_geometric
from torch_geometric.transforms import Compose

qm9_target_dict = {
    0: "mu",
    1: "alpha",
    2: "homo",
    3: "lumo",
    4: "gap",
    5: "r2",
    6: "zpve",
    7: "U0",
    8: "U",
    9: "H",
    10: "G",
    11: "Cv",
}


class QM9(QM9_geometric):
    """
    QM9 dataset wrapper for PyTorch Geometric QM9 dataset.
    
    This class extends the PyTorch Geometric QM9 dataset to provide additional
    functionality for working with specific molecular properties.
    """

    mu = "mu"
    alpha = "alpha"
    homo = "homo"
    lumo = "lumo"
    gap = "gap"
    r2 = "r2"
    zpve = "zpve"
    U0 = "U0"
    U = "U"
    H = "H"
    G = "G"
    Cv = "Cv"

    available_properties = [
        mu,
        alpha,
        homo,
        lumo,
        gap,
        r2,
        zpve,
        U0,
        U,
        H,
        G,
        Cv,
    ]

    def __init__(
        self,
        root: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        dataset_arg=None,
    ):
        """
        Initialize the QM9 dataset.
        
        Args:
            root (str): Root directory where the dataset should be saved.
            transform: Transform to be applied to each data object. If None,
                       defaults to _filter_label.
            pre_transform: Transform to be applied to each data object before saving.
            pre_filter: Function that takes in a data object and returns a boolean,
                       indicating whether the item should be included.
            dataset_arg (str): The property to train on. Must be one of the available
                              properties defined in qm9_target_dict.
        
        Raises:
            AssertionError: If dataset_arg is None.
        """
        assert dataset_arg is not None, (
            "Please pass the desired property to "
            'train on via "dataset_arg". Available '
            f'properties are {", ".join(qm9_target_dict.values())}.'
        )

        self.label = dataset_arg
        label2idx = dict(zip(qm9_target_dict.values(), qm9_target_dict.keys(), strict=False))
        self.label_idx = label2idx[self.label]

        if transform is None:
            transform = self._filter_label
        else:
            transform = Compose([transform, self._filter_label])

        super(QM9, self).__init__(
            root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )


    @staticmethod
    def label_to_idx(label: str) -> int:
        """
        Convert a property label to its corresponding index.
        
        Args:
            label (str): The property label to convert.
            
        Returns:
            int: The index corresponding to the property label.
        """
        label2idx = dict(zip(qm9_target_dict.values(), qm9_target_dict.keys(), strict=False))
        return label2idx[label]

    def mean(self, divide_by_atoms: bool = True) -> float:
        """
        Calculate the mean of the target property across the dataset.
        
        Args:
            divide_by_atoms (bool): Whether to normalize the property by the number
                                   of atoms in each molecule.
            
        Returns:
            float: The mean value of the target property.
        """
        if not divide_by_atoms:
            get_labels = lambda i: self.get(i).y
        else:
            get_labels = lambda i: self.get(i).y/self.get(i).pos.shape[0]

        y = torch.cat([get_labels(i) for i in range(len(self))], dim=0)
        assert len(y.shape) == 2
        if y.shape[1] != 1:
            y = y[:, self.label_idx]
        else:
            y = y[:, 0]
        return y.mean(axis=0)
    def min(self, divide_by_atoms: bool = True) -> float:
        """
        Calculate the minimum of the target property across the dataset.
        
        Args:
            divide_by_atoms (bool): Whether to normalize the property by the number
                                   of atoms in each molecule.
            
        Returns:
            float: The minimum value of the target property.
        """
        if not divide_by_atoms:
            get_labels = lambda i: self.get(i).y
        else:
            get_labels = lambda i: self.get(i).y/self.get(i).pos.shape[0]

        y = torch.cat([get_labels(i) for i in range(len(self))], dim=0)
        assert len(y.shape) == 2
        if y.shape[1] != 1:
            y = y[:, self.label_idx]
        else:
            y = y[:, 0]
        return y.min(axis=0)

    def std(self, divide_by_atoms: bool = True) -> float:
        """
        Calculate the standard deviation of the target property across the dataset.
        
        Args:
            divide_by_atoms (bool): Whether to normalize the property by the number
                                   of atoms in each molecule.
            
        Returns:
            float: The standard deviation of the target property.
        """
        if not divide_by_atoms:
            get_labels = lambda i: self.get(i).y
        else:
            get_labels = lambda i: self.get(i).y/self.get(i).pos.shape[0]

        y = torch.cat([get_labels(i) for i in range(len(self))], dim=0)
        assert len(y.shape) == 2
        if y.shape[1] != 1:
            y = y[:, self.label_idx]
        else:
            y = y[:, 0]
        return y.std(axis=0)

    def get_atomref(self, max_z: int = 100) -> torch.Tensor:
        """
        Get atomic reference values for the target property.
        
        Args:
            max_z (int): Maximum atomic number to consider.
            
        Returns:
            torch.Tensor: Tensor of atomic reference values, or None if not available.
        """
        atomref = self.atomref(self.label_idx)
        if atomref is None:
            return None
        if atomref.size(0) != max_z:
            tmp = torch.zeros(max_z).unsqueeze(1)
            idx = min(max_z, atomref.size(0))
            tmp[:idx] = atomref[:idx]
            return tmp
        return atomref

    def _filter_label(self, batch) -> torch.Tensor:
        """
        Filter the batch to only include the target property.
        
        Args:
            batch: A batch of data from the dataset.
            
        Returns:
            torch.Tensor: The filtered batch with only the target property.
        """
        batch.y = batch.y[:, self.label_idx].unsqueeze(1)
        return batch
