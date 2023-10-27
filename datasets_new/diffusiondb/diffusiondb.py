from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
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
from json import load, dump
from os.path import join, basename

class DiffusionDBFineTuneDataset(Dataset):
    def __init__(self, split='train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train',caption_data='part-000001',backbone='t5-small'):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode

        self.n_boxes = 36
        self.use_vision = True
        self.backbone = backbone
        self.do_lower_case = False
        self.max_text_length = 20
        self.max_n_boxes = 36
        self.caption_data = caption_data
        self.gen_max_length = 20

        # Loading datasets to data
        self.source = split
        if self.verbose:
            print('Data source: ', self.source)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.backbone,
            max_length=self.max_text_length,
            do_lower_case=self.do_lower_case)
        data =[]
        
        for i in range(20):
            cur_json_path = join(self.raw_dataset,'part-{num:06d}'.format(num=i+1),'part-{num:06d}.json'.format(num=i+1))

            data.extend(load(open(cur_json_path, "r", encoding="utf8")).items())

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
        out_dict['args'] = self.args

        img_name,datum = self.data[idx]
        img_path = join(self.raw_dataset, img_name)
        prompt = datum["p"]

        return {
            'answer': prompt,
            'image': img_path,
            'type': 'image,answer',
        }

    @staticmethod
    def collate_fn(batch):
        return batch


def fetch_data(data_path, subset_size, split="train", args=None):
    # DataLoader
    dataset = DiffusionDBFineTuneDataset(split='train', raw_dataset=data_path,topk = subset_size)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,  # load torch.utils.data.Dataset or torchvision.datasets
        batch_size=256,
        shuffle=False,
        drop_last=True,  # if div batch_size is not zero，False means that the last batch will be smaller，True means we drop the last small batch
        collate_fn=DiffusionDBFineTuneDataset.collate_fn,
        num_workers=0
    )
    return [loader]