from pathlib import Path
from torch.utils.data import DataLoader, Dataset

import common.io_utils as io


class NYUv2Dataset(Dataset):
    def __init__(self, root_path, split="train"):
        self.data_path = Path(root_path)
        self.subset = split
        self._load_dataset()
    
    def _load_dataset(self):
        if self.subset == 'train':
            index_dir = self.data_path / "sync" / "index_train.txt"
            img_dir = self.data_path / "sync"
            with index_dir.open("r") as f:
                index_nyuv2 = f.readlines()
        else:
            img_dir = self.data_path / "labelled"
            index_dir = self.data_path / "sync" / "index_test.txt"
            with index_dir.open("r") as f:
                index_nyuv2 = f.readlines()

        self.image_paths = []
        self.depth_paths = []
        for filename_pair in index_nyuv2:
            image_name, depth_name = filename_pair.split()[:2]
            self.image_paths.append(str(img_dir / image_name))
            self.depth_paths.append(str(img_dir / depth_name))

    def __getitem__(self, i):
        image_path, depth_path = self.image_paths[i], self.depth_paths[i]
        return {
            "image": image_path,
            "depth": depth_path
        }
    
    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def collate_fn(batch):
        return batch


def fetch_data(data_path, subset_size, split="train"):
    dataset = NYUv2Dataset(root_path=data_path, split=split)
    loader = DataLoader(
        dataset=dataset,
        batch_size=subset_size,
        shuffle=False,
        drop_last=True,
        collate_fn=NYUv2Dataset.collate_fn,
        num_workers=0
    )
    return loader