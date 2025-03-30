import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_warn


def train_val_test_split(
    dset_len: int,
    train_size: float or int or None,
    val_size: float or int or None,
    test_size: float or int or None,
    seed: int,
) -> tuple:
    """
    Split dataset indices into training, validation, and test sets.
    
    This function splits a dataset of length dset_len into training, validation,
    and test sets according to the specified sizes. The sizes can be specified as
    fractions of the dataset (float) or as absolute counts (int).
    
    Args:
        dset_len (int): Total length of the dataset.
        train_size (float or int or None): Size of the training set. If float, interpreted
            as a fraction of the dataset. If int, interpreted as an absolute count.
            If None, calculated as the remainder after val_size and test_size.
        val_size (float or int or None): Size of the validation set. If float, interpreted
            as a fraction of the dataset. If int, interpreted as an absolute count.
            If None, calculated as the remainder after train_size and test_size.
        test_size (float or int or None): Size of the test set. If float, interpreted
            as a fraction of the dataset. If int, interpreted as an absolute count.
            If None, calculated as the remainder after train_size and val_size.
        seed (int): Random seed for reproducibility.
        
    Returns:
        tuple: A tuple containing three numpy arrays (idx_train, idx_val, idx_test)
            with the indices for each split.
            
    Raises:
        AssertionError: If more than one of train_size, val_size, test_size is None,
            or if any split size is negative, or if the total split size exceeds
            the dataset length.
    """
    assert (train_size is None) + (val_size is None) + (test_size is None) <= 1, "Only one of train_size, val_size, test_size is allowed to be None."

    is_float = (isinstance(train_size, float), isinstance(val_size, float), isinstance(test_size, float))

    train_size = round(dset_len * train_size) if is_float[0] else train_size
    val_size = round(dset_len * val_size) if is_float[1] else val_size
    test_size = round(dset_len * test_size) if is_float[2] else test_size

    if train_size is None:
        train_size = dset_len - val_size - test_size
    elif val_size is None:
        val_size = dset_len - train_size - test_size
    elif test_size is None:
        test_size = dset_len - train_size - val_size

    # Adjust split sizes if they exceed the dataset length
    if train_size + val_size + test_size > dset_len:
        if is_float[2]:
            test_size -= 1
        elif is_float[1]:
            val_size -= 1
        elif is_float[0]:
            train_size -= 1

    assert train_size >= 0 and val_size >= 0 and test_size >= 0, (
        f"One of training ({train_size}), validation ({val_size}) or "
        f"testing ({test_size}) splits ended up with a negative size."
    )

    total = train_size + val_size + test_size
    assert dset_len >= total, f"The dataset ({dset_len}) is smaller than the combined split sizes ({total})."

    if total < dset_len:
        rank_zero_warn(f"{dset_len - total} samples were excluded from the dataset")

    # Generate random indices
    idxs = np.arange(dset_len, dtype=np.int64)
    idxs = np.random.default_rng(seed).permutation(idxs)

    # Split indices into train, validation, and test sets
    idx_train = idxs[:train_size]
    idx_val = idxs[train_size: train_size + val_size]
    idx_test = idxs[train_size + val_size: total]

    return np.array(idx_train), np.array(idx_val), np.array(idx_test)


def make_splits(
    dataset_len: int,
    train_size: float or int or None,
    val_size: float or int or None,
    test_size: float or int or None,
    seed: int,
    filename: str = None,
    splits: str = None,
) -> tuple:
    """
    Create or load dataset splits and optionally save them to a file.
    
    This function either loads existing splits from a file or creates new splits
    using train_val_test_split. The resulting splits can be saved to a file.
    
    Args:
        dataset_len (int): Total length of the dataset.
        train_size (float or int or None): Size of the training set. See train_val_test_split.
        val_size (float or int or None): Size of the validation set. See train_val_test_split.
        test_size (float or int or None): Size of the test set. See train_val_test_split.
        seed (int): Random seed for reproducibility.
        filename (str, optional): If provided, the splits will be saved to this file.
        splits (str, optional): If provided, splits will be loaded from this file
            instead of being generated.
            
    Returns:
        tuple: A tuple containing three torch tensors (idx_train, idx_val, idx_test)
            with the indices for each split.
    """
    if splits is not None:
        splits = np.load(splits)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
        idx_test = splits["idx_test"]
    else:
        idx_train, idx_val, idx_test = train_val_test_split(
            dataset_len,
            train_size,
            val_size,
            test_size,
            seed,
        )

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    return torch.from_numpy(idx_train), torch.from_numpy(idx_val), torch.from_numpy(idx_test)


class MissingLabelException(Exception):
    """
    Exception raised when a required label is missing from the dataset.
    
    This exception is used to indicate that a required label or property
    is not available in the dataset being processed.
    """
    pass
