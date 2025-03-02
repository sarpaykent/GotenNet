from abc import ABC, abstractmethod
from os.path import join
from typing import Dict, List, Optional, Tuple, Union

import torch
from lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from torch_geometric.loader import DataLoader
from torch_scatter import scatter
from tqdm import tqdm

from src.utils import RankedLogger
from .components.md17 import MD17
from .components.md22 import MD22
from .components.qm9 import QM9
from .components.rmd17 import rMD17
from .components.utils import MissingLabelException, make_splits

log = RankedLogger(__name__, rank_zero_only=True)


def normalize_positions(batch: torch.Tensor) -> torch.Tensor:
    """Normalize the positions of atoms in a batch."""
    center = batch.center_of_mass
    batch.pos = batch.pos - center
    return batch


class BaseDatasetPreparation(ABC):
    """Abstract base class for dataset preparation."""

    def __init__(self, hparams: Dict):
        self.hparams = hparams

    @abstractmethod
    def load_dataset(self, root: str, dataset_arg: str) -> Union[MD17, QM9]:
        """Load the dataset."""
        pass

    @abstractmethod
    def split_dataset(self, dataset_size: int, train_size: int, val_size: int, seed: int, splits=None) -> Tuple[
        List[int], List[int], List[int]]:
        """Split the dataset into train, validation, and test sets."""
        pass


class rMD17Preparation(BaseDatasetPreparation):
    """Dataset preparation for rMD17."""

    def load_dataset(self, root: str, dataset_arg: str) -> rMD17:
        return rMD17(root=root, dataset_arg=dataset_arg)

    def split_dataset(self, dataset_size: int, train_size: int, val_size: int, seed: int, splits=None) -> Tuple[
        List[int], List[int], List[int]]:
        dataset = self.load_dataset(self.hparams["dataset_root"], dataset_arg=self.hparams["dataset_arg"])

        if splits is not None:
            split_idxs = dataset.get_split(splits)
            assert len(split_idxs) == 2, "Expected two splits"
            assert len(split_idxs[
                           0]) == train_size + val_size, f"Expected train+val{train_size + val_size} size != {len(split_idxs[0])} size"
            # idx_train, idx_val = split_idxs[0][:train_size], split_idxs[0][train_size:]
            idx_train, idx_val, _ = make_splits(
                len(split_idxs[0]),
                train_size,
                val_size,
                None,
                seed,
                join(self.hparams["output_dir"], "splits.npz"),
                splits=None,
            )
            idx_test = split_idxs[1]
            print(f"[ID: {splits}] train {len(idx_train)}, val {len(idx_val)}, test {len(idx_test)}")
        else:
            idx_train, idx_val, idx_test = make_splits(
                len(dataset),
                train_size,
                val_size,
                None,
                seed,
                join(self.hparams["output_dir"], "splits.npz"),
                splits,
            )

        return idx_train, idx_val, idx_test


class MD22Preparation(BaseDatasetPreparation):
    """Dataset preparation for MD22."""

    def load_dataset(self, root: str, dataset_arg: str) -> MD22:
        return MD22(root=root, dataset_arg=dataset_arg)

    def split_dataset(self, dataset_size: int, train_size: int, val_size: int, seed: int, splits=None) -> Tuple[
        List[int], List[int], List[int]]:
        dataset = self.load_dataset(self.hparams["dataset_root"], dataset_arg=self.hparams["dataset_arg"])
        train_val_size = dataset.molecule_splits[self.hparams["dataset_arg"]]
        train_size = round(train_val_size * 0.95)
        val_size = train_val_size - train_size

        idx_train, idx_val, idx_test = make_splits(
            len(dataset),
            train_size,
            val_size,
            None,
            seed,
            join(self.hparams["output_dir"], "splits.npz"),
            splits,
        )

        print(f"train {len(idx_train)}, val {len(idx_val)}, test {len(idx_test)}")
        return idx_train, idx_val, idx_test


class QM9Preparation(BaseDatasetPreparation):
    """Dataset preparation for QM9."""

    def load_dataset(self, root: str, dataset_arg: str) -> QM9:
        transform = normalize_positions if self.hparams["normalize_positions"] is None else None
        if transform:
            log.warning("Normalizing positions for QM9 dataset.")
        return QM9(root=root, dataset_arg=dataset_arg, transform=transform)

    def split_dataset(self, dataset_size: int, train_size: int, val_size: int, seed: int, splits=None) -> Tuple[
        List[int], List[int], List[int]]:
        from sklearn.model_selection import train_test_split

        all_indices = list(range(dataset_size))
        train_val, test = train_test_split(all_indices, test_size=0.1, random_state=seed)
        train, val = train_test_split(train_val, train_size=train_size / (train_size + val_size), random_state=seed)

        return train, val, test


class DataModule(LightningDataModule):
    """Main data module for handling datasets."""

    def __init__(self, hparams: Union[Dict, "DictConfig"]):
        super().__init__()
        self.hparams.update(hparams)
        self._mean: Optional[float] = None
        self._std: Optional[float] = None
        self._saved_dataloaders: Dict[str, DataLoader] = {}
        self.dataset: Optional[Union[MD17, QM9]] = None
        self.loaded: bool = False
        self.idx_train: Optional[List[int]] = None
        self.idx_val: Optional[List[int]] = None
        self.idx_test: Optional[List[int]] = None
        self.train_dataset: Optional[Union[MD17, QM9]] = None
        self.val_dataset: Optional[Union[MD17, QM9]] = None
        self.test_dataset: Optional[Union[MD17, QM9]] = None

        self.dataset_preparations = {
            "rMD17": rMD17Preparation(self.hparams),
            "MD22": MD22Preparation(self.hparams),
            "QM9": QM9Preparation(self.hparams)
        }

    def get_metadata(self, label: Optional[str] = None) -> Dict:
        """Get metadata for the dataset."""
        if label is not None:
            self.hparams["dataset_arg"] = label

        if not self.loaded:
            self.prepare_dataset()
            self.loaded = True

        return {
            'atomref': self.atomref,
            'dataset': self.dataset,
            'mean': self.mean,
            'std': self.std
        }

    def prepare_dataset(self):
        """Prepare the dataset for use."""
        dataset_type = self.hparams["dataset"]
        if dataset_type not in self.dataset_preparations:
            raise ValueError(f"Dataset {dataset_type} not supported")

        preparation = self.dataset_preparations[dataset_type]
        self.dataset = preparation.load_dataset(self.hparams["dataset_root"], self.hparams["dataset_arg"])
        self.idx_train, self.idx_val, self.idx_test = preparation.split_dataset(
            len(self.dataset),
            self.hparams["train_size"],
            self.hparams["val_size"],
            self.hparams["seed"],
            self.hparams['splits'],
        )

        print(f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}")
        self.train_dataset = self.dataset[self.idx_train]
        self.val_dataset = self.dataset[self.idx_val]
        self.test_dataset = self.dataset[self.idx_test]

        if self.hparams["standardize"]:
            self._standardize()

    def train_dataloader(self) -> DataLoader:
        """Return the train dataloader."""
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """Return the validation dataloader(s)."""
        loaders = [self._get_dataloader(self.val_dataset, "val")]
        delta = 1 if self.hparams['reload'] == 1 else 2
        if self.hparams["test_interval"] != 0 and (
                len(self.test_dataset) > 0
                and (self.trainer.current_epoch + delta) % self.hparams["test_interval"] == 0
        ):
            loaders.append(self._get_dataloader(self.test_dataset, "test"))
        return loaders

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return self._get_dataloader(self.test_dataset, "test")

    @property
    def atomref(self) -> Optional[torch.Tensor]:
        """Return the atom reference if available."""
        if hasattr(self.dataset, "get_atomref"):
            return self.dataset.get_atomref()
        return None

    @property
    def mean(self) -> Optional[float]:
        """Return the mean of the dataset."""
        return self._mean

    @property
    def std(self) -> Optional[float]:
        """Return the standard deviation of the dataset."""
        return self._std

    def _get_dataloader(self, dataset: Union[MD17, QM9], stage: str, store_dataloader: bool = True) -> DataLoader:
        """Get a dataloader for the specified stage."""
        store_dataloader = (store_dataloader and not self.hparams["reload"])
        if stage in self._saved_dataloaders and store_dataloader:
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.hparams["batch_size"]
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.hparams["inference_batch_size"]
            shuffle = False
        else:
            raise ValueError(f"Invalid stage: {stage}")

        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
        )

        if store_dataloader:
            self._saved_dataloaders[stage] = dl
        return dl

    @rank_zero_only
    def _standardize(self):
        """Standardize the dataset."""

        def get_label(batch: torch.Tensor, atomref: Optional[torch.Tensor]) -> Tuple[
            torch.Tensor, Optional[torch.Tensor]]:
            if batch.y is None:
                raise MissingLabelException()

            dy = batch.dy.squeeze().clone() if 'dy' in batch else None

            if atomref is None:
                return batch.y.clone(), dy

            atomref_energy = scatter(atomref[batch.z], batch.batch, dim=0)
            return (batch.y.squeeze() - atomref_energy.squeeze()).clone(), dy

        if self.hparams["standardize"] == 2:
            label = self.hparams["dataset_arg"]
            log.warning(f"Overall standardization is used for label {label}.")

            self._mean = self.train_dataset.mean()
            self._std = self.train_dataset.std()
            return

        data = tqdm(
            self._get_dataloader(self.train_dataset, "val", store_dataloader=False),
            desc="computing mean and std",
        )
        try:
            atomref = self.atomref if self.hparams["prior_model"] == "Atomref" else None
            ys = [get_label(batch, atomref) for batch in data]
            ys, _ = zip(*ys)
            ys = torch.cat(ys)
        except MissingLabelException:
            rank_zero_warn(
                "Standardize is true but failed to compute dataset mean and "
                "standard deviation. Maybe the dataset only contains forces."
            )
            return

        self._mean = ys.mean(dim=0)[0].item()
        self._std = ys.std(dim=0)[0].item()
        print(f"mean: {self._mean}, std: {self._std}")
