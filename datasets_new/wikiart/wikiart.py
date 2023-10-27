import os
import random
import pandas as pd
from torch.utils.data.dataset import Dataset
import torch


class WiKiArtDataset(Dataset):
    def __init__(self, root_path, img_path, split='train' ,transform=None):
        self.img_path = img_path
        self.transform = transform
        self.root_path = root_path
        self.split = split

        # root_files = img_path
        self.artist_list = pd.read_csv(os.path.join(root_path, f'artist_{split}.csv'))
        self.style_list = pd.read_csv(os.path.join(root_path, f'style_{split}.csv'))
        self.genre_list = pd.read_csv(os.path.join(root_path, f'genre_{split}.csv'))
        # self.subject_list = pd.read_csv(os.path.join(root_path, 'subject.csv'))

        temp = [i.strip('\n').split(' ') for i in open(os.path.join(root_path, 'artist_class.txt'), 'r').readlines()]
        self.artist_id_class = dict(zip([int(i[0]) for i in temp],[i[1].replace("_", " ") for i in temp]))
        temp = [i.split(' ') for i in open(os.path.join(root_path, 'style_class.txt'), 'r').readlines()]
        self.style_id_class = dict(zip([int(i[0]) for i in temp],[i[1].strip('\n').replace("_", " ") for i in temp]))
        temp = [i.split(' ') for i in open(os.path.join(root_path, 'genre_class.txt'), 'r').readlines()]
        self.genre_id_class = dict(zip([int(i[0]) for i in temp],[i[1].strip('\n').replace("_", " ") for i in temp]))

    def __getitem__(self, index):
        img_name = os.path.join(self.img_path, self.artist_list.loc[index, 'Images'])  # image path
        artist_name = self.artist_id_class[self.artist_list.loc[index, 'Labels']]
        style_name = self.style_id_class[self.style_list.loc[index, 'Labels']]
        genre_name = self.genre_id_class[self.genre_list.loc[index, 'Labels']]
        # subject_name = self.subject_list[self.subject_list.loc[index, 'Labels']]
        artist = torch.tensor(self.artist_list.loc[index, 'Labels'])  # get label
        style = torch.tensor(self.style_list.loc[index, 'Labels'])  # get label
        genre = torch.tensor(self.genre_list.loc[index, 'Labels'])  # get label
        # subject = torch.tensor(self.subject_list.loc[index, 'labels'])  # get label
        answer = f"This artwork is in the genre of {genre_name}, created by {artist_name}, and exemplifies the style of {style_name}."

        return {
            'manifest_filename': img_name,
            'manifest_root': self.root_path,
            'artist_index': artist,
            'style_index': style,
            'genre_index': genre,
            'artist_name': artist_name,
            'style_name': style_name,
            'genre_name': genre_name,
            'image': img_name,
            'answer':answer,
            'type': 'image,answer',
        }

    def __len__(self):
        return len(self.artist_list)

    @staticmethod
    def collate_fn(batch):
        return batch


def fetch_data(data_path, subset_size, split="train"):
    # DataLoader
    splits = ['train','val']
    loaders=[]
    for sp in splits:
        dataset = WiKiArtDataset(root_path=data_path, img_path=os.path.join(data_path, 'img'),split=sp)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,  # load torch.utils.data.Dataset or torchvision.datasets
            batch_size=500,
            shuffle=False,
            drop_last=True,  # if div batch_size is not zero，False means that the last batch will be smaller，True means we drop the last small batch
            collate_fn=WiKiArtDataset.collate_fn,
            num_workers=0
        )
        loaders.append(loader)
    return loaders


if __name__ == "__main__":
    pass