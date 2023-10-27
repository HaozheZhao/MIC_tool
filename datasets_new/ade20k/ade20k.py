from pathlib import Path
from torch.utils.data import DataLoader, Dataset


class ADE20kDataset(Dataset):
    def __init__(self, root_path, split="train"):
        super().__init__()
        self.index_dir = Path(root_path)
        self.img_dir = Path(root_path) / "images"
        self.anno_dir = Path(root_path) / "annotations"
        self.subset = split
        self._load_dataset()
    
    def _load_dataset(self):
        self.samples = [x.rstrip() for x in (self.info.index_dir / f'ADE20K_object150_{self.subset}.txt').open('r')]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        filename = self.samples[i]
        img_name = str(self.img_dir / filename)
        seg_name = str(self.anno_dir / filename).replace(".jpg", ".png")
        # image, segmentation
        return {
            "image": img_name,
            "segmentation": seg_name
        }

    @staticmethod
    def collate_fn(batch):
        return batch


def fetch_data(data_path, subset_size, split="train"):
    dataset = ADE20kDataset(root_path=data_path, split=split)
    loader = DataLoader(
        dataset=dataset,
        batch_size=subset_size,
        shuffle=False,
        drop_last=True,
        collate_fn=ADE20kDataset.collate_fn,
        num_workers=0
    )
    return loader