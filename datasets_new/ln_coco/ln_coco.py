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


dataset_dir = "./data"

class LNCOCOCaptionFineTuneDataset(Dataset):
    def read_jsonl_tolist(self,path):
        with open(path,'r') as f:
            lines = f.readlines()
        return [json.loads(line) for line in lines]

    def __init__(self, data_path='./data/ln_coco',split='train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None,
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

        with open(join(dataset_dir,'ADE20K_2016_07_26/name2path.json'),'r') as f:
            self.ade_name2path = json.load(f)

        self.source = split
        if self.verbose:
            print('Data source: ', self.source)
        self.train_json =[ 'ade20k_train_captions.jsonl','coco_train_captions.jsonl','flickr30k_train_captions.jsonl']
        self.val_json =[ 'ade20k_validation_captions.jsonl','coco_val_captions.jsonl','flickr30k_val_captions.jsonl']
        data =[]

        if split == 'train':
            for json_dir in self.train_json :
                data.extend(
                    self.read_jsonl_tolist(join(self.data_path,json_dir)))
        else:
            for json_dir in self.val_json :
                data.extend(
                    self.read_jsonl_tolist(join(self.data_path,json_dir)))



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


        img_id = datum['image_id']
        dataset = datum['dataset_id']
        if 'ADE' in dataset:
            img_cur = join(dataset_dir,self.ade_name2path[img_id+'.jpg'])

        elif 'mscoco' in dataset:
            image_id_str = str(img_id).zfill(12) + '.jpg'
            coco_path = dataset.split('_')[1]
            img_cur = join(self.data_path,coco_path,image_id_str)

        elif 'Flick30k' in dataset:
            flickr_path = "flickr/flickr30k-images"
            img_cur = join(dataset_dir,flickr_path,img_id+ '.jpg')
        
        else:
            raise ValueError(f'Unknown dataset {dataset}')
        cap = [datum['caption'].replace('this image','image 0').replace('this picture','image 0'), datum['caption']]
        caption = cap[ random.randint(0,1)]
        return {
            'answer': caption,
            'image': img_cur,
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
        dataset = LNCOCOCaptionFineTuneDataset(data_path=data_path,split = split , topk = subset_size)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,  # load torch.utils.data.Dataset or torchvision.datasets
            batch_size=256,
            shuffle=False,
            drop_last=True,  # if div batch_size is not zero，False means that the last batch will be smaller，True means we drop the last small batch
            collate_fn=LNCOCOCaptionFineTuneDataset.collate_fn,
            num_workers=0
        )
        loaders.append(loader)
    return loaders