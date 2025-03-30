from os.path import join
from typing import Any, Dict, Optional, Union

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from torch_geometric.loader import DataLoader
from torch_scatter import scatter
from tqdm import tqdm

from gotennet import utils

from .components.qm9 import QM9
from .components.utils import MissingLabelException, make_splits

log = utils.get_logger(__name__)


def normalize_positions(batch):
    """
    Normalize positions by subtracting center of mass.
    
    Args:
        batch: Data batch with position information.
        
    Returns:
        batch: Batch with normalized positions.
    """
    center = batch.center_of_mass
    batch.pos = batch.pos - center
    return batch


class DataModule(LightningDataModule):
    """
    DataModule for handling various molecular datasets.
    
    This class provides a unified interface for loading, splitting, and
    standardizing different types of molecular datasets.
    """

    def __init__(self, hparams: Union[Dict, Any]):
        """
        Initialize the DataModule with configuration parameters.
        
        Args:
            hparams: Hyperparameters for the datamodule.
        """
        # Check if hparams is omegaconf.dictconfig.DictConfig
        if type(hparams) == "omegaconf.dictconfig.DictConfig":
            hparams = dict(hparams)
        super(DataModule, self).__init__()
        hparams = dict(hparams)

        # Update hyperparameters
        if hasattr(hparams, "__dict__"):
            self.hparams.update(hparams.__dict__)
        else:
            self.hparams.update(hparams)

        # Initialize attributes
        self._mean, self._std = None, None
        self._saved_dataloaders = dict()
        self.dataset = None
        self.loaded = False

    def get_metadata(self, label: Optional[str] = None) -> Dict:
        """
        Get metadata about the dataset.
        
        Args:
            label: Optional label to set as dataset_arg.
            
        Returns:
            Dict containing dataset metadata.
        """
        if label is not None:
            self.hparams["dataset_arg"] = label

        if self.loaded == False:
            self.prepare_dataset()
            self.loaded = True

        return {
            'atomref': self.atomref,
            'dataset': self.dataset,
            'mean': self.mean,
            'std': self.std
        }

    def prepare_dataset(self):
        """
        Prepare the dataset for training, validation, and testing.
        
        Loads the appropriate dataset based on the configuration and
        creates the train/val/test splits.
        
        Raises:
            AssertionError: If the specified dataset type is not supported.
        """
        dataset_type = self.hparams['dataset']

        # Validate dataset type is supported
        assert hasattr(self, f"_prepare_{dataset_type}"), \
            f"Dataset {dataset_type} not defined"

        # Call the appropriate dataset preparation method
        dataset_preparer = lambda t: getattr(self, f"_prepare_{t}")()
        self.idx_train, self.idx_val, self.idx_test = dataset_preparer(dataset_type)

        log.info(f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}")

        # Set up dataset subsets
        self.train_dataset = self.dataset[self.idx_train]
        self.val_dataset = self.dataset[self.idx_val]
        self.test_dataset = self.dataset[self.idx_test]

        # Standardize if requested
        if self.hparams["standardize"]:
            self._standardize()

    def train_dataloader(self):
        """
        Get the training dataloader.
        
        Returns:
            DataLoader for training data.
        """
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        """
        Get the validation dataloader.
        
        Returns:
            DataLoader for validation data.
        """
        return self._get_dataloader(self.val_dataset, "val")

    def test_dataloader(self):
        """
        Get the test dataloader.
        
        Returns:
            DataLoader for test data.
        """
        return self._get_dataloader(self.test_dataset, "test")

    @property
    def atomref(self):
        """
        Get atom reference values if available.
        
        Returns:
            Atom reference values or None.
        """
        if hasattr(self.dataset, "get_atomref"):
            return self.dataset.get_atomref()
        return None

    @property
    def mean(self):
        """
        Get mean value for standardization.
        
        Returns:
            Mean value.
        """
        return self._mean

    @property
    def std(self):
        """
        Get standard deviation value for standardization.
        
        Returns:
            Standard deviation value.
        """
        return self._std

    def _get_dataloader(
        self,
        dataset,
        stage: str,
        store_dataloader: bool = True
    ):
        """
        Create a dataloader for the given dataset and stage.
        
        Args:
            dataset: The dataset to create a dataloader for.
            stage: The stage ('train', 'val', or 'test').
            store_dataloader: Whether to store the dataloader for reuse.
            
        Returns:
            DataLoader for the dataset.
        """
        store_dataloader = (store_dataloader and not self.hparams["reload"])
        if stage in self._saved_dataloaders and store_dataloader:
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.hparams["batch_size"]
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.hparams["inference_batch_size"]
            shuffle = False

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
        """
        Standardize the dataset by computing mean and standard deviation.
        
        This method computes the mean and standard deviation of the dataset
        for standardization purposes. It handles different standardization
        approaches based on the configuration.
        """
        def get_label(batch, atomref):
            """
            Extract label from batch, accounting for atom references if provided.
            """
            if batch.y is None:
                raise MissingLabelException()

            dy = None
            if 'dy' in batch:
                dy = batch.dy.squeeze().clone()

            if atomref is None:
                return batch.y.clone(), dy

            atomref_energy = scatter(atomref[batch.z], batch.batch, dim=0)
            return (batch.y.squeeze() - atomref_energy.squeeze()).clone(), dy

        # Standard approach: compute mean and std from data
        data = tqdm(
            self._get_dataloader(self.train_dataset, "val", store_dataloader=False),
            desc="computing mean and std",
        )
        try:
            atomref = self.atomref if self.hparams.get("prior_model") == "Atomref" else None
            ys = [get_label(batch, atomref) for batch in data]
            # Convert array with n elements and each element contains 2 values
            # to array of two elements with n values
            ys, dys = zip(*ys, strict=False)
            ys = torch.cat(ys)
        except MissingLabelException:
            rank_zero_warn(
                "Standardize is true but failed to compute dataset mean and "
                "standard deviation. Maybe the dataset only contains forces."
            )
            return None

        self._mean = ys.mean(dim=0)[0].item()
        self._std = ys.std(dim=0)[0].item()
        log.info(f"mean: {self._mean}, std: {self._std}")

    def _prepare_QM9(self):
        """
        Load and prepare the QM9 dataset with appropriate splits.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                Indices for train, validation, and test splits.
        """
        # Apply position normalization if requested
        transform = normalize_positions if self.hparams["normalize_positions"] else None
        if transform:
            log.warning("Normalizing positions.")

        self.dataset = QM9(
            root=self.hparams["dataset_root"],
            dataset_arg=self.hparams["dataset_arg"],
            transform=transform
        )

        train_size = self.hparams["train_size"]
        val_size = self.hparams["val_size"]

        idx_train, idx_val, idx_test = make_splits(
            len(self.dataset),
            train_size,
            val_size,
            None,
            self.hparams["seed"],
            join(self.hparams["output_dir"], "splits.npz"),
            self.hparams["splits"],
        )

        return idx_train, idx_val, idx_test
