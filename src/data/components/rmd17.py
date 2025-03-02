import os
import os.path as osp

import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_warn
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_tar
from tqdm import tqdm


class rMD17(InMemoryDataset):
    revised_url = ('https://figshare.com/ndownloader/files/23950376')

    molecule_files = dict(
        aspirin='rmd17_aspirin.npz',
        azobenzene='rmd17_azobenzene.npz',
        benzene='rmd17_benzene.npz',
        ethanol='rmd17_ethanol.npz',
        malonaldehyde='rmd17_malonaldehyde.npz',
        naphthalene='rmd17_naphthalene.npz',
        paracetamol='rmd17_paracetamol.npz',
        salicylic='rmd17_salicylic.npz',
        toluene='rmd17_toluene.npz',
        uracil='rmd17_uracil.npz',
    )

    available_molecules = list(molecule_files.keys())

    def __init__(self, root, transform=None, pre_transform=None, dataset_arg=None):
        assert dataset_arg is not None, (
            "Please provide the desired comma separated molecule(s) through"
            f"'dataset_arg'. Available molecules are {', '.join(rMD17.available_molecules)} "
            "or 'all' to train on the combined dataset."
        )

        if dataset_arg == "all":
            dataset_arg = ",".join(rMD17.available_molecules)
        self.molecules = dataset_arg.split(",")

        if len(self.molecules) > 1:
            rank_zero_warn(
                "MD17 molecules have different reference energies, "
                "which is not accounted for during training."
            )

        super(rMD17, self).__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    def len(self):
        return self.data.y.size(0)

    @property
    def raw_file_names(self):
        return [osp.join('rmd17', 'npz_data', rMD17.molecule_files[mol]) for mol in self.molecules]

    def get_split(self, idx):
        assert idx in [0, 1, 2, 3, 4]

        sets = ['index_train', 'index_test']

        out = []
        for set_name in sets:
            split_path = osp.join(self.root, 'raw', 'rmd17', 'splits', set_name + f'_0{idx + 1}.csv')
            # check file exists
            if not osp.exists(split_path):
                raise FileNotFoundError(f"File {split_path} not found")
            # load csv
            with open(split_path, 'r') as f:
                split = [int(line.strip()) for line in f.readlines()]
            out.append(split)
        return out

    @property
    def processed_file_names(self):
        return [f"rmd17-{mol}.pt" for mol in self.molecules]

    def download(self):
        path = download_url(self.revised_url, self.raw_dir)
        extract_tar(path, self.raw_dir, mode='r:bz2')
        os.unlink(path)

    def process(self):
        for path, processed_path in zip(self.raw_paths, self.processed_paths):
            data_npz = np.load(path)
            z = torch.from_numpy(data_npz["nuclear_charges"]).long()
            positions = torch.from_numpy(data_npz["coords"]).float()
            energies = torch.from_numpy(data_npz["energies"]).float()
            forces = torch.from_numpy(data_npz["forces"]).float()
            energies.unsqueeze_(1)

            samples = []
            for pos, y, dy in tqdm(zip(positions, energies, forces), total=energies.size(0)):

                data = Data(z=z, pos=pos, y=y.unsqueeze(1), dy=dy)

                if self.pre_filter is not None:
                    data = self.pre_filter(data)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                samples.append(data)

            data, slices = self.collate(samples)
            torch.save((data, slices), processed_path)
