import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import common.io_utils as io
import pandas as pd
from os.path import join ,exists
import json
from  model.blip2.processing_blip_2 import Blip2Processor
from PIL import Image
import random
class RefCOCODataset(Dataset):
    def __init__(self, root_path, split="train",dataset='refcoco', splitBy='unc', topk=-1):
        super().__init__()
        self.DATA_DIR = join(root_path, dataset)
        self.IMAGE_DIR = join("./data/coco", 'train2014')
        self.topk=topk
        ref_file = join(self.DATA_DIR, 'refs('+splitBy+').p')
        self.data = {}
        self.data['dataset'] = dataset
        self.data['refs'] = pd.read_pickle(ref_file)
        instances_file = join(self.DATA_DIR, 'instances.json')
        instances = json.load(open(instances_file, 'r'))
        self.data['images'] = instances['images']
        self.data['annotations'] = instances['annotations']
        self.data['categories'] = instances['categories']
        self.subset = split

        self.createIndex()
        if topk>0:
            self.data['refs'] = self.data['refs'][:self.topk]
 


    def createIndex(self):
        # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        for ann in self.data['annotations']:
            Anns[ann['id']] = ann
            imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
        for img in self.data['images']:
            Imgs[img['id']] = img
        for cat in self.data['categories']:
            Cats[cat['id']] = cat['name']

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        for ref in self.data['refs']:
            # ids
            ref_id = ref['ref_id']
            ann_id = ref['ann_id']
            category_id = ref['category_id']
            image_id = ref['image_id']

            # add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = Anns[ann_id]
            annToRef[ann_id] = ref

            # add mapping of sent
            for sent in ref['sentences']:
                Sents[sent['sent_id']] = sent
                sentToRef[sent['sent_id']] = ref
                sentToTokens[sent['sent_id']] = sent['tokens']

        # create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens
    
    def __len__(self):
        return len( self.data['refs'])

    def cut_bbox_and_save(self,file,bbox):
        img_path = "_".join(file.split("_")[:-1])+".jpg"
        img = Image.open(img_path)
        if not exists(file):
            left = bbox[0]
            top = bbox[1]
            width = bbox[2]
            height = bbox[3]
            bbox = (left, top, left+width,top+height)
            bbox=tuple(bbox)
            newim=img.crop(bbox)
            newim.save(file)
        quadrant = self.get_bbox_quadrant(bbox,img)
        return img_path,quadrant
    def get_bbox_quadrant(self,bbox, image):
        # We need to first extract the coordinates of the bbox
        # Assuming the bbox is in the format [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = bbox

        # We can then calculate the width and height of the bbox
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        # We can also get the width and height of the image
        image_width, image_height = image.size

        # We can then calculate the center coordinates of the bbox
        bbox_center_x = x_min + bbox_width / 2
        bbox_center_y = y_min + bbox_height / 2

        # We can then calculate the relative position of the bbox center within the image
        relative_bbox_center_x = bbox_center_x / image_width
        relative_bbox_center_y = bbox_center_y / image_height

        # We can then determine which quadrant of the image the bbox center is in
        if relative_bbox_center_x < 0.33 and relative_bbox_center_y < 0.33:
            return "top left"
        elif relative_bbox_center_x >= 0.33 and relative_bbox_center_x < 0.67 and relative_bbox_center_y < 0.33:
            return "top middle"
        elif relative_bbox_center_x >= 0.67 and relative_bbox_center_y < 0.33:
            return "top right"
        elif relative_bbox_center_x < 0.33 and relative_bbox_center_y >= 0.33 and relative_bbox_center_y < 0.67:
            return "middle left"
        elif relative_bbox_center_x >= 0.33 and relative_bbox_center_x < 0.67 and relative_bbox_center_y >= 0.33 and relative_bbox_center_y < 0.67:
            return "center"
        elif relative_bbox_center_x >= 0.67 and relative_bbox_center_y >= 0.33 and relative_bbox_center_y < 0.67:
            return "middle right"
        elif relative_bbox_center_x < 0.33 and relative_bbox_center_y >= 0.67:
            return "bottom left"
        elif relative_bbox_center_x >= 0.33 and relative_bbox_center_x < 0.67 and relative_bbox_center_y >= 0.67 and relative_bbox_center_y <= 1:
            return "bottom middle"
        else:
            return "bottom right"

    def __getitem__(self, i):
        sample =  self.data['refs'][i]
        cur_img = sample['file_name']
        image_path = join(self.IMAGE_DIR, cur_img)
        bbox = self.Anns[sample['ann_id']]['bbox']
        big_image_path,quadrant = self.cut_bbox_and_save(image_path,bbox)

        sents = [each['sent'] for each in sample['sentences']]

        re = [{"image": big_image_path,
            "answer": sent,
            "quadrant": quadrant} for sent in sents]
        return re


    @staticmethod
    def collate_fn(batch):
        return batch


def fetch_data(data_path, subset_size, split="train"):
    dataset = RefCOCODataset(root_path=data_path, split=split,topk=subset_size)
    loader = DataLoader(
        dataset=dataset,
        batch_size=256,
        shuffle=False,
        drop_last=True,
        collate_fn=RefCOCODataset.collate_fn,
        num_workers=0
    )
    return [loader]