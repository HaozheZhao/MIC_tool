
from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import random
from multiprocessing import Pool
import pickle
import math
from os.path import join
from tqdm import tqdm
import torch
import numpy as np
from copy import deepcopy
from os.path import join,relpath

from torch.utils.data.distributed import DistributedSampler

VIDEO_PATH='./data/funqa/'
class FUNQAFineTuneDataset(Dataset):
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

        self.path_map = {
            'C':"creative",
            'H':'humor',
            'M':'magic'
        }
        if self.verbose:
            print("# all sentences:", len(self.data))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        re =[]
        datum = self.data[idx]

        ###### Image ######
        source = self.split
        video_path = datum['visual_input']
        v_type = self.path_map[video_path[0]]
        cur_video = join(VIDEO_PATH,f"{source}/{source}_{v_type}/{video_path}")
        out_dict['image'] =cur_video 
        out_dict['question'] = datum['instruction']
        out_dict['answer'] = datum['output']
        return out_dict


    def collate_fn(self, batch):
        return batch



def get_loader(data_path, args=None, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1):

    verbose = (gpu == 0)

    _dset = FUNQA2Dataset(data_path,split, verbose)

    dataset = FUNQAFineTuneDataset(
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

    loader.task = 'FUNQA'

    return loader


class FUNQA2Dataset:

    def read_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        return papers

    def __init__(self, FUNQA_dir,splits: str, verbose=True):
        self.name = splits
        self.splits = splits.split(',')[0]

        # Loading datasets to data
        js = self.read_json(join(FUNQA_dir,f"annotation_with_ID/funqa_{self.splits}.json"))
        
        ignore = ['H1','C1','M1']
        self.data =  [ each for each in js if each['task'] not in ignore]


        if verbose:
            print("Load %d data from split(s) %s." %
                  (len(self.data), self.name))



    def __len__(self):
        return len(self.data)


def fetch_data(data_path, subset_size, split="train", args=None):
    # DataLoader
    loaders = []
    for split in ['train','test'] :

        data_loader = get_loader(
            split=split,
            data_path =data_path,
            mode = split,
            batch_size=256,  
            workers=4,
            distributed=False, gpu=0,topk=subset_size)
        loaders.append(data_loader)
    return loaders

