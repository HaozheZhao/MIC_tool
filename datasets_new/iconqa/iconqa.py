import os
from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import random
from multiprocessing import Pool
import pickle
import math
import pandas as pd
from os.path import join
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
from copy import deepcopy
from os.path import join,relpath

from torch.utils.data.distributed import DistributedSampler

IMG_PATH='./data/iconqa/img/'
class ICONQAFineTuneDataset(Dataset):
    def __init__(self, split='train', data_path ='./data/' ,raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train'):
        super().__init__()

        self.data_path = data_path
        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode

        # Loading datasets to data
        self.split = split
        if self.verbose:
            print('Data source: ', self.split)

        data = self.raw_dataset.data

        if topk > 0:
            data = data[:topk]
            if self.verbose:
                print(f"Use only {topk} data")

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank

        self.data = data

        if self.verbose:
            print("# all sentences:", len(self.data))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        re =[]
        instance_folder = self.data[idx]
        d_path = join(self.data_path, self.split,'choose_txt')
        instance_path = os.path.join(d_path, instance_folder)
        if os.path.isdir(instance_path):  
            data_file = os.path.join(instance_path, "data.json")
            
            with open(data_file, "r") as f:
                instance_data = json.load(f)
            
            choices = instance_data["choices"]
            images = join(d_path,f"{instance_folder}/image.png")
            out_dict['question'] = instance_data["question"]
            out_dict['image'] = images
            answer = instance_data["answer"]
            out_dict['answer'] = f"option {answer+1}: {choices[answer]}"
            # input_txt = [prefix_txt+f" Given the image 0 and question, does the image {i+1}: <image{i+1}>图 is the answer for the following question: {question} Yes or No?" for i in range(len(choices))]
            out_dict['options'] = ". ".join([ f"option {i+1}: {choices[i]}" for i in range(len(choices))])+'.'
            return out_dict
        return None


    def collate_fn(self, batch):
        return batch



def get_loader(data_path, args=None, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1):

    verbose = (gpu == 0)

    _dset = ICONQA2Dataset(data_path,split, verbose)

    dataset = ICONQAFineTuneDataset(
        split,
        data_path =data_path,
        raw_dataset=_dset,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        mode=mode)

    if distributed and mode == 'train':
        train_sampler = DistributedSampler(dataset)
    else:
        train_sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=True, sampler=train_sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,
            sampler=None,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    loader.task = 'ICONQA'

    return loader


class ICONQA2Dataset:

    def read_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        return papers

    def __init__(self, ICONQA_dir,splits: str, verbose=True):
        self.name = splits
        self.splits = splits.split(',')[0]

        # Loading datasets to data
# choose_txt
        DATA_PATH = join(ICONQA_dir, self.splits,'choose_txt')
        self.data = os.listdir(DATA_PATH)

        if verbose:
            print("Load %d data from split(s) %s." %
                  (len(self.data), self.name))



    def __len__(self):
        return len(self.data)


def fetch_data(data_path, subset_size, split="train", args=None):
    # DataLoader
    loaders = []
    for split in ['train','val','test'] :
        data_loader = get_loader(
            split=split,
            data_path =data_path,
            mode = split,
            batch_size=256,  
            workers=4,
            distributed=False, gpu=0,topk=subset_size)
        loaders.append(data_loader)
    return loaders

