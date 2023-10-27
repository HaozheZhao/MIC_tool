import os
from os.path import join
from tqdm import tqdm
from PIL import Image
# crop vcr dataset images

base_dir = '.data/vcr/'
newimg_dir = 'vcr2images_slices'
img_dir = join(base_dir,'vcr1images')
new_dir = join(base_dir,newimg_dir)
os.makedirs(new_dir,exist_ok=True)
data_type = ['train','val']
import json
path='.data/vcr/train.jsonl'
def readjsonl(path):
    file = open(path, 'r', encoding='utf-8')
    papers = []
    for line in file.readlines():
        dic = json.loads(line)
        papers.append(dic)
    file.close()
    return papers
for types in data_type:
    infos = readjsonl(join(base_dir,f"{types}.jsonl"))
    print(f"precessing total {len(infos)} annotations")
    json_infos=[]
    for info in tqdm(infos):
        img_fn = info['img_fn']
        metadata_fn = info['metadata_fn']
        image = Image.open(join(img_dir,img_fn))
        with open(join(img_dir,metadata_fn),'r') as f:
            js = json.load(f)
        bboxs = js['boxes']
        dir_name, png_name = img_fn.split('/')
        file_dir = join(new_dir,dir_name)
        os.makedirs(file_dir,exist_ok=True)
        page = png_name.split('.')
        file_type = page[-1]
        name = '.'.join(page[:-1])
        bbox_paths =[]
        for i,bbox in enumerate(bboxs):
            left = bbox[0]
            top = bbox[1]
            width = bbox[2]
            height = bbox[3]
            temp_bbox = (left, top, width,height)
            new_im = image.crop(temp_bbox)
            bbox_paths.append(join(newimg_dir,dir_name,f'{name}_{i}.{file_type}'))
            new_im.save(join(file_dir,f'{name}_{i}.{file_type}'))
        info['bbox_path'] = bbox_paths
    with open(join(base_dir,f'cliped_vcr_{types}.json'),'w') as f:
        json.dump(infos,f)
            



            


