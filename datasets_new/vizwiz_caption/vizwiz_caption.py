from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import random
from multiprocessing import Pool
import h5py
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
from copy import deepcopy
import os
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from os.path import join


class VIZWIZCaptionCaptionFineTuneDataset(Dataset):
    def __init__(self, data_path='./data/vizwiz_caption',split='train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None,
                 mode='train'):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args
        self.data_path = data_path
        self.mode = mode

        self.n_boxes = 36
        self.use_vision = True
        self.backbone = 't5-small'
        self.do_lower_case = False
        self.max_text_length = 20
        self.max_n_boxes = 36
        self.gen_max_length = 20

        # Loading datasets to data
        self.source = split
        if self.verbose:
            print('Data source: ', self.source)



        data_info_path = join(self.data_path , f'annotations/{self.source}.json')

        with open(data_info_path) as f:
            re = json.load(f)
        self.id2img = { each['id']:each['file_name'] for each in re['images'] }
        data = re['annotations']
        n_images = len(data)
       
        if self.verbose:
            print(f"{self.source} has {n_images} images")
            print(f"Loaded {len(data)} data from", split)

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank
        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        self.data = data

        if self.verbose:
            print("# all sentences:", len(self.data))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        datum = self.data[idx]
        img_name = self.id2img[datum['image_id']]
        cur_img = join(self.data_path , f'{self.source}/{img_name}')

        return {
            'answer': datum['caption'],
            'image': cur_img,
            'type': 'image,answer',
        }

    @staticmethod
    def collate_fn(batch):
        return batch

data_path = None
img_path=None

def fetch_data(data_path, subset_size, split="train", args=None):
    # DataLoader
    loaders = []
    splits = ['train','val']
    for split in splits:
        dataset = VIZWIZCaptionCaptionFineTuneDataset(data_path=data_path,split = split , topk = subset_size)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,  # load torch.utils.data.Dataset or torchvision.datasets
            batch_size=256,
            shuffle=False,
            drop_last=True,  # if div batch_size is not zero，False means that the last batch will be smaller，True means we drop the last small batch
            collate_fn=VIZWIZCaptionCaptionFineTuneDataset.collate_fn,
            num_workers=0
        )
        loaders.append(loader)
    return loaders