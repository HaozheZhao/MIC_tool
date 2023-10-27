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


class TEXTCAPSCaptionFineTuneDataset(Dataset):
    def __init__(self, data_path='./data/textcaps',split='train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None,
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
        self.caption_data = f"captions_{split}2014"
        self.gen_max_length = 20

        # Loading datasets to data
        self.source = split
        if self.verbose:
            print('Data source: ', self.source)
        def load_json(path):
            with open(path) as f:
                return json.load(f)
        if split == 'train':
            json_data = load_json(join(self.data_path, 'TextCaps_0.1_train.json'))
        elif split == 'val':
            json_data = load_json(join(self.data_path, 'TextCaps_0.1_val.json'))
        else:
            raise NotImplementedError
        data = json_data['data']

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


        datum = self.data[idx]
        image_id = datum['image_id']
        cur_img_path = join(self.data_path, 'train_images', image_id+'.jpg')
        return {
            'answer': datum['caption_str'],
            'image': cur_img_path,
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
        dataset = TEXTCAPSCaptionFineTuneDataset(data_path=data_path,split = split , topk = subset_size)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,  # load torch.utils.data.Dataset or torchvision.datasets
            batch_size=256,
            shuffle=False,
            drop_last=True,  # if div batch_size is not zero，False means that the last batch will be smaller，True means we drop the last small batch
            collate_fn=TEXTCAPSCaptionFineTuneDataset.collate_fn,
            num_workers=0
        )
        loaders.append(loader)
    return loaders