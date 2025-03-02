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

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, dataset_arg=None):
        assert dataset_arg is not None, (
            "Please pass the desired property to "
            'train on via "dataset_arg". Available '
            f'properties are {", ".join(qm9_target_dict.values())}.'
        )

        self.label = dataset_arg
        label2idx = dict(zip(qm9_target_dict.values(), qm9_target_dict.keys()))
        self.label_idx = label2idx[self.label]

        if transform is None:
            transform = self._filter_label
        else:
            transform = Compose([transform, self._filter_label])

        super(QM9, self).__init__(root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

    @staticmethod
    def label_to_idx(label):
        label2idx = dict(zip(qm9_target_dict.values(), qm9_target_dict.keys()))
        return label2idx[label]

    def mean(self, divide_by_atoms=True) -> float:
        if not divide_by_atoms:
            get_labels = lambda i: self.get(i).y
        else:
            get_labels = lambda i: self.get(i).y / self.get(i).pos.shape[0]

        y = torch.cat([get_labels(i) for i in range(len(self))], dim=0)
        assert len(y.shape) == 2
        print(y.shape, "yshape!")
        if y.shape[1] != 1:
            y = y[:, self.label_idx]
        else:
            y = y[:, 0]
        return y.mean(axis=0)

    def min(self, divide_by_atoms=True) -> float:
        if not divide_by_atoms:
            get_labels = lambda i: self.get(i).y
        else:
            get_labels = lambda i: self.get(i).y / self.get(i).pos.shape[0]

        y = torch.cat([get_labels(i) for i in range(len(self))], dim=0)
        assert len(y.shape) == 2
        if y.shape[1] != 1:
            y = y[:, self.label_idx]
        else:
            y = y[:, 0]
        return y.min(axis=0)

    def std(self, divide_by_atoms=True) -> float:
        if not divide_by_atoms:
            get_labels = lambda i: self.get(i).y
        else:
            get_labels = lambda i: self.get(i).y / self.get(i).pos.shape[0]

        y = torch.cat([get_labels(i) for i in range(len(self))], dim=0)
        assert len(y.shape) == 2
        if y.shape[1] != 1:
            y = y[:, self.label_idx]
        else:
            y = y[:, 0]
        return y.std(axis=0)

    def get_atomref(self, max_z=100):
        atomref = self.atomref(self.label_idx)
        if atomref is None:
            return None
        if atomref.size(0) != max_z:
            tmp = torch.zeros(max_z).unsqueeze(1)
            idx = min(max_z, atomref.size(0))
            tmp[:idx] = atomref[:idx]
            return tmp
        return atomref

    def _filter_label(self, batch):
        batch.y = batch.y[:, self.label_idx].unsqueeze(1)
        return batch
