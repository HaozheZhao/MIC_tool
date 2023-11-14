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

IMG_PATH='./data/coco/train2014/'
class LLaVAFineTuneDataset(Dataset):
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
        datum = self.data[idx]

        ###### Image ######
        source = self.split
        img_path = datum['image']
        cur_img = join(IMG_PATH,"COCO_train2014_"+img_path)
        conversation = datum['conversations']
        start = conversation[0]
        end = conversation[-1]
        answer=end['value'].strip()
        conversation = conversation[1:-1]
        context=["User: "+start['value'].replace('<image>','').strip()]
        for con in conversation:
            subject=con['from']
            prefix="User: " if subject =='human' else "Bot: "
            text = prefix+con['value'].strip()
            context.append(text)
        out_dict['image'] =cur_img 
        out_dict['question'] = "\n".join(context)
        out_dict['answer'] = answer
        return [out_dict]


    def collate_fn(self, batch):
        return batch



def get_loader(data_path, args=None, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1):

    verbose = (gpu == 0)

    _dset = LLaVA2Dataset(data_path,split, verbose)

    dataset = LLaVAFineTuneDataset(
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

    loader.task = 'LLaVA'

    return loader


class LLaVA2Dataset:

    def read_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        return papers

    def __init__(self, LLaVA_dir,splits: str, verbose=True):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data = []
        data_json_list=['complex_reasoning_77k.json','conversation_58k.json','detail_23k.json','llava_instruct_80k.json']
        for split in self.splits:
            for json_dir in data_json_list:
                self.data.extend(
                    self.read_json(join(LLaVA_dir,json_dir)))




        if verbose:
            print("Load %d data from split(s) %s." %
                  (len(self.data), self.name))



    def __len__(self):
        return len(self.data)


def fetch_data(data_path, subset_size, split="train", args=None):
    # DataLoader
    loaders = []
    for split in ['train'] :
        data_loader = get_loader(
            split=split,
            data_path =data_path,
            mode = split,
            batch_size=256,  
            workers=4,
            distributed=False, gpu=0,topk=subset_size)
        loaders.append(data_loader)
    return loaders

