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
import re
import os
from torch.utils.data.distributed import DistributedSampler

import transformers

from transformers import AutoTokenizer
from os.path import join 


class GQAFineTuneDataset(Dataset):
    def __init__(self, split='train,valid',  data_path ='./data/',raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train'):
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

        # Loading datasets to data
        self.sources = split.split(',')
        if self.verbose:
            print('Data sources: ', self.sources)


        data = list(raw_dataset.data.items())

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
        question_id, datum = self.data[idx]


        ###### Image ######
        if self.use_vision:
            img_id = datum['imageId']
            cur_source = join(self.data_path,'images')
            cur_img = join(cur_source,img_id+".jpg")
            out_dict['image'] = cur_img

        ###### Text #####

        out_dict['question_id'] = question_id

        out_dict['question'] = datum['question']

        out_dict['answer'] = datum['fullAnswer']
        out_dict['type'] = 'image,question,answer'
        return out_dict

    @staticmethod
    def collate_fn(batch):
        return batch

def get_loader(args=None, data_path='./data/gqa',split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1, verbose=None):
    if verbose is None:
        verbose = (gpu == 0)

    _dset = GQADataset(split, data_path,verbose)

    dataset = GQAFineTuneDataset(
        split,
        data_path =data_path,
        raw_dataset=_dset,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        mode=mode)

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None
    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    loader.task = 'gqa'

    return loader


class GQADataset:

    def __init__(self, splits: str, data_path='./data/gqa', verbose=True):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            self.data.append(json.load(open(join(data_path, f"{split}_balanced_questions.json" ))))
        if len(self.data) == 1:
            self.data = self.data[0]
        if verbose:
            print("Load %d data from split(s) %s." %
                  (len(self.data), self.name))

        # List to dict (for evaluation and others)

    def __len__(self):
        return len(self.data)


def fetch_data(data_path, subset_size, split="train", args=None):
    # DataLoader
    data_path = data_path
    loaders = []
    for split in ['train','val','testdev'] :
        data_loader = get_loader(
            split=split,
            data_path =data_path,
            mode= split,
            batch_size=256,  
            workers=4,
            distributed=False, gpu=0,topk=-1)
        loaders.append(data_loader)
    return loaders