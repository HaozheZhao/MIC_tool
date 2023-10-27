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
project_dir = Path(__file__).resolve().parent.parent
workspace_dir = project_dir.parent
dataset_dir = workspace_dir.joinpath('data/').resolve()
flickr_dir = dataset_dir.joinpath('flickr')
vg_dir = dataset_dir.joinpath('vg')
flickr_img_dir = flickr_dir.joinpath('flickr30k-images/')
flickr_feature_dir = flickr_dir.joinpath('features')


class COCOCaptionFineTuneDataset(Dataset):
    def __init__(self, split='karpathy_train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train'):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode

        self.n_boxes = 36
        self.use_vision = True
        self.backbone = 't5-small'
        self.do_lower_case = False
        self.max_text_length = 20
        self.max_n_boxes = 36
        self.caption_data = 'dataset_flickr30k'
        self.gen_max_length = 20

        # Loading datasets to data
        self.source = split
        if self.verbose:
            print('Data source: ', self.source)


        self.tokenizer = AutoTokenizer.from_pretrained(
            self.backbone,
            max_length=self.max_text_length,
            do_lower_case=self.do_lower_case)

        data_info_path = dataset_dir.joinpath(f'flickr/{self.caption_data}.json')
        with open(data_info_path) as f:
            karpathy_data = json.load(f)

        split_rename = {
            'train': 'train',
            'restval': 'train',
            'val': 'val',
            'test': 'test'
        }

        n_images = 0

        data = []
        img_path = os.path.join(self.raw_dataset, 'flickr30k-images/')
        for datum in karpathy_data['images']:
            re_split = split_rename[datum['split']]
            if re_split != self.source.split('_')[-1]:
                continue

            if re_split == 'train':
                for d in datum['sentences']:
                    # img_id = datum['filename'].split('.')[0]
                    img_id = str(os.path.join(img_path,datum['filename']))

                    new_datum = {
                        'img_id': img_id,
                        'sent': d['raw'].strip(),
                        'targets': [d['raw'].strip() for d in datum['sentences']],
                        'is_train': True,
                    }
                    data.append(new_datum)
            else:
                for d in datum['sentences']:
                    img_id =  str(os.path.join(img_path,datum['filename']))
                    new_datum = {
                        'img_id': img_id,
                        'sent': d['raw'].strip(),
                        'targets': [d['raw'].strip() for d in datum['sentences']],
                        # 'targets': "##".join([d['raw'].strip() for d in datum['sentences']]),


                        'is_train': False,
                    }
                    data.append(new_datum)

            n_images += 1

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

        self.source_to_h5 = {}

        if self.max_n_boxes == 36:
            self.source_to_h5.update({
                'all': flickr_dir.joinpath('features').joinpath('flickr30k_boxes36.h5'),
            })


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]

        if self.use_vision:
            img_id = datum['img_id']
            out_dict['img_id'] = img_id


            n_boxes = self.max_n_boxes

            feats = np.zeros(shape=(n_boxes, 2048), dtype=np.float32)
            feats = torch.from_numpy(feats)

            n_boxes = min(n_boxes, self.max_n_boxes)
            out_dict['n_boxes'] = n_boxes
            feats = feats[:n_boxes]
            out_dict['vis_feats'] = feats

            prefix = ''



            input_tokens = [prefix]

            input_text = ' '.join(input_tokens)

            input_ids = self.tokenizer.encode(
                input_text,
                max_length=self.max_text_length, truncation=True)

        out_dict['input_text'] = input_text

        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        if datum['is_train']:
            sent = datum['sent'].strip()
            target_ids = self.tokenizer.encode(sent, max_length=self.gen_max_length, truncation=True)

            assert len(target_ids) <= self.gen_max_length, len(target_ids)
            out_dict['sent'] = sent
            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)
        else:
            out_dict['sent'] = datum['sent'].strip()
        if 'targets' in datum:
            out_dict['targets'] = datum['targets']

        # return out_dict
        if datum['is_train']:
            return {
                'answer': out_dict['sent'],
                'image': out_dict['img_id'],
                'type': 'image,answer',
            }
        else:
            return {
                'answer': out_dict['sent'],
                'image': out_dict['img_id'],
                'type': 'image,answer',
            }
            

    @staticmethod
    def collate_fn(batch):
        return batch


def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1):

    verbose = (gpu == 0)

    dataset = COCOCaptionFineTuneDataset(
        split,
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

    if verbose:
        loader.evaluator = COCOCaptionEvaluator()

    loader.task = 'caption'

    return loader



class COCOCaptionEvaluator:
    def __init__(self):
        import language_evaluation
        self.evaluator = language_evaluation.CocoEvaluator(verbose=False)


    def evaluate(self, predicts, answers):

        results = self.evaluator.run_evaluation(predicts, answers)

        return results

data_path = None
img_path=None

def fetch_data(data_path, subset_size, split="train", args=None):
    # DataLoader
    data_path = data_path
    img_path = os.path.join(data_path, 'img')
    split = ['train','val','test']
    loaders = []
    for sp in split:
        dataset = COCOCaptionFineTuneDataset(raw_dataset = data_path , split = sp,topk=subset_size)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,  # load torch.utils.data.Dataset or torchvision.datasets
            batch_size=256,
            shuffle=False,
            drop_last=True,  # if div batch_size is not zero，False means that the last batch will be smaller，True means we drop the last small batch
            collate_fn=COCOCaptionFineTuneDataset.collate_fn,
            num_workers=0
        )
        loaders.append(loader)
    return loaders