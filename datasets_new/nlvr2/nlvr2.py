from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import random
from multiprocessing import Pool
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
from copy import deepcopy
from os.path import join,relpath

from torch.utils.data.distributed import DistributedSampler

project_dir = Path(__file__).resolve().parent.parent
workspace_dir = project_dir.parent
dataset_dir = workspace_dir.joinpath('data/').resolve()
nlvr_dir = dataset_dir.joinpath('nlvr2')
nlvr_feature_dir = nlvr_dir.joinpath('features')



class NLVRFineTuneDataset(Dataset):
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

        datum = self.data[idx]

        ###### Image ######
        source = self.split
        cur_source = join(self.data_path,'features')
        cur_source = join(cur_source,source)
        if source =='train':
            directory = datum['directory']
            cur_source = join(cur_source,str(directory))
  
        temp = datum['identifier'].split('-')
        if temp[-1] == "0":
            img_0 = "-".join(temp[:-1])+'-img'+temp[-1]+'.png'
            img_1 = "-".join(temp[:-1])+'-img1.png'
        else:
            img_0 = "-".join(temp[:-1])+'-img0.png'
            img_1 = "-".join(temp[:-1])+'-img'+temp[-1]+'.png'
        cur_img_0 = join(cur_source,img_0)
        cur_img_1 = join(cur_source,img_1)
        # out_dict['image'] =f"{cur_img_0}###{cur_img_1}"
        out_dict['image'] =[cur_img_0,cur_img_1]

        ###### Text #####

        out_dict['question'] = datum['sentence']
        out_dict['answer'] = datum['label']
        out_dict['options'] = "\n".join(['True','False'])



        return out_dict


    def collate_fn(self, batch):
        return batch



def get_loader(data_path, args=None, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1):

    verbose = (gpu == 0)

    _dset = NLVR2Dataset(split, verbose)

    dataset = NLVRFineTuneDataset(
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

    loader.task = 'nlvr'

    return loader


class NLVR2Dataset:
    """
    An NLVR2 data example in json file:
    {
        "identifier": "train-10171-0-0",
        "img0": "train-10171-0-img0",
        "img1": "train-10171-0-img1",
        "label": 0,
        "sent": "An image shows one leather pencil case, displayed open with writing implements tucked inside.",
        "uid": "nlvr2_train_0"
    }
    """
    def read_json(self, path):
        file = open(path, 'r', encoding='utf-8')
        papers = []
        for line in file.readlines():
            dic = json.loads(line)
            papers.append(dic)
        file.close()
        return papers

    def __init__(self, splits: str, verbose=True):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            self.data.extend(
                self.read_json(nlvr_dir.joinpath(f'{split}.json')))
        if verbose:
            print("Load %d data from split(s) %s." %
                  (len(self.data), self.name))



    def __len__(self):
        return len(self.data)


def fetch_data(data_path, subset_size, split="train", args=None):
    # DataLoader
    loaders = []
    for split in ['train','dev','test1'] :
        data_loader = get_loader(
            split=split,
            data_path =data_path,
            mode= split,
            batch_size=256,  
            workers=4,
            distributed=False, gpu=0,topk=-1)
        loaders.append(data_loader)
    return loaders

