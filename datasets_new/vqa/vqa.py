
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
import re
import os
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer



class VQADataset(Dataset):
    def __init__(self, split='train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train', root_path=None, img_path=None):
        super().__init__()

        self.dataset_dir = root_path
        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        # self.args = args
        self.n_boxes = 36
        self.use_vision = True
        self.do_lower_case = False
        self.max_text_length = 20

        self.mode = mode
        # Loading datasets to data
        self.sources = split.split(',')
        if self.verbose:
            print('Data sources: ', self.sources)


        self.img_ids_to_source = {}
        data_info_dicts = []
        for source in self.sources:
            data_info_path = os.path.join(self.dataset_dir, f'{source}.json')
            with open(data_info_path) as f:
                _data_info_dicts = json.load(f)
                for _d in _data_info_dicts:
                    if 'train2014' in _d['img_id']:
                        self.img_ids_to_source[_d['img_id']] = 'train2014'
                    elif 'val2014' in _d['img_id']:
                        self.img_ids_to_source[_d['img_id']] = 'val2014'
                    elif 'test2015' in _d['img_id']:
                        self.img_ids_to_source[_d['img_id']] = 'test2015'
                        _d['source'] = source

                data_info_dicts.extend(_data_info_dicts)
            if self.verbose:
                print(f"Loaded {len(_data_info_dicts)} data from", source)

        data = data_info_dicts

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

        ###### Image ######
        if self.use_vision:
            img_id = datum['img_id']
            cur_img = datum['img_id']+'.jpg'
            source = self.img_ids_to_source[img_id]
            
            img_path = os.path.join(self.dataset_dir,source)
            out_dict['img_id'] = os.path.join(img_path,cur_img)

        ###### Text #####
        if 'sent' in datum:
            sent = datum['sent']
        elif 'question' in datum:
            sent = datum['question']

        question_id = datum['question_id']
        out_dict['question_id'] = question_id


        out_dict['sent'] = sent

        if 'is_topk_optimal' in datum:
            out_dict['is_topk_optimal'] = datum['is_topk_optimal']

        if 'label' in datum:
            label = datum['label']
            out_dict['label'] = label

            answers = []
            scores = []
            for a, s in label.items():
                answers.append(a)
                scores.append(s)

            score_sum = sum(scores)

            if score_sum == 0:
                answer = ''
                score = 0.
            else:
                prob = [score / score_sum for score in scores]
                choice = np.random.multinomial(1, prob).argmax()
                answer = answers[choice]
                score = scores[choice]
                assert len(answer) > 0, (sent, label, choice, answer)

            out_dict['answer'] = answer
            out_dict['score'] = score
            out_dict['all_answers'] = answers

        return {
            'question': out_dict['sent'],
            'image': out_dict['img_id'],
            'answer': out_dict['answer'],
            'type': 'image,question,answer',
        }

    @staticmethod
    def collate_fn(batch):
        return batch

def get_loader(args, split='karpathy_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0, topk=-1,root_path=None):

    verbose = (gpu == 0)

    _dset = VQAData(root_path,split, verbose)

    dataset = VQADataset(
        split,
        raw_dataset=_dset,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        mode=mode,
        root_path = root_path)

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

    loader.task = 'vqa'

    return loader


class VQAData:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """

    def __init__(self, dataset_dir,splits: str, verbose=True):
        self.name = splits
        self.splits = splits.split(',')

        with open(os.path.join(dataset_dir, 'v2_mscoco_train2014_annotations.json')) as f:
            train2014_data = json.load(f)
        with open(os.path.join(dataset_dir,'v2_mscoco_val2014_annotations.json')) as f:
            val2014_data = json.load(f)
        train2014_id2datum = {}
        for datum in train2014_data['annotations']:
            qid = datum['question_id']
            train2014_id2datum[qid] = datum
        val2014_id2datum = {}
        for datum in val2014_data['annotations']:
            qid = datum['question_id']
            val2014_id2datum[qid] = datum
        self.id2datum_gt = {**train2014_id2datum, **val2014_id2datum}

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(
                json.load(open(os.path.join(dataset_dir,"%s.json" % split))))

        if verbose:
            print("Load %d data from split(s) %s." %
                  (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Topk Answers
        self.ans2label = json.load(
            open(os.path.join(dataset_dir,"trainval_ans2label.json")))
        self.label2ans = json.load(
            open(os.path.join(dataset_dir,"trainval_label2ans.json")))
        assert len(self.ans2label) == len(self.label2ans)


    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)

def fetch_data(data_path, subset_size, split="train", args=None):
    # DataLoader
    loaders = []
    modes = {"train":"train","karpathy_val":'val',"karpathy_test":'test'}
    for split in ['train','karpathy_val','karpathy_test']:
        loader = get_loader(args, split=split, mode=modes[split],
               batch_size=256, workers=4, distributed=False, gpu=0, topk=subset_size,root_path = data_path)
        loaders.append(loader)
    return loaders