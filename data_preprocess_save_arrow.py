import base64
import json
from os.path import join
import jsonlines
from tqdm import tqdm
import transformers
from model.blip2 import Blip2Processor
from model.blip2 import Blip2Config
from model.instructblip import InstructBlipProcessor
from PIL import Image
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import h5py
import fsspec
import copy
import pickle as pk
from glob import glob
from os.path import join
import threading
from datasets.arrow_writer import ArrowWriter                                                                                                                                                   
import os
from moviepy.editor import VideoFileClip
from PIL import Image
from random import sample
import random
from tqdm import tqdm

import json
from random import sample
import os
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
def getsample_vcr(path,types,split_num = -1,store_path =''):
    with open(path,'r') as f:
        js = json.load(f)
    result =[]
    for each in tqdm(js['instances']):
        inputs= each['input']
        outputs = each['output']
        input_text = "User: "+each['input'][0]['text'] 
        if len(each['input']) ==3:
            input_image = [each['input'][1]['image']] +each['input'][2]['bbox_list']
        else:
            input_image = each['input'][1]['image'] 
        if len(input_image) > 10:
            continue
        output_text = "Bot: "+each['output'][0]['answer'] 
        output_image = ""
        js_dict ={'input_text':input_text,'input_image':input_image,'output_text':output_text,'output_image':output_image}
        result.append(js_dict)
    if split_num == -1:
        with open(os.path.join(store_path,f'vcr-promot-1-{types}_nature.json'),'w') as f:
            for dic in result:
                f.write(json.dumps(dic, ensure_ascii=False)+"\n") 
    else:
        with open(os.path.join(store_path,f'vcr-promot-1-{types}_sample_{split_num}_nature.json'),'w') as f:
            for dic in sample(result,split_num):
                f.write(json.dumps(dic, ensure_ascii=False)+"\n")

def getsample_llava(path,types,split_num = -1,store_path =''):
    with open(path,'r') as f:
        js = json.load(f)
    result =[]
    for each in tqdm(js['instances']):
        inputs= each['input']
        outputs = each['output']
        input_text = each['input'][0]['text'] 
        input_image = each['input'][1]['image'] 
        output_text = "Bot: "+each['output'][0]['answer'] 
        output_image = ""
        js_dict ={'input_text':input_text,'input_image':input_image,'output_text':output_text,'output_image':output_image}
        result.append(js_dict)
    if split_num == -1:
        with open(os.path.join(store_path,f'llava-promot-1-{types}_nature.json'),'w') as f:
            for dic in result:
                f.write(json.dumps(dic, ensure_ascii=False)+"\n") 
    else:
        with open(os.path.join(store_path,f'llava-promot-1-{types}_sample_{split_num}_nature.json'),'w') as f:
            for dic in sample(result,split_num):
                f.write(json.dumps(dic, ensure_ascii=False)+"\n")


def qa_form_fewshot(instances,text,image,sample_num,nature,out_text,output_answer='answer'):
    poss_instances= sample(instances,random.randint(0,sample_num))
    images=[]
    texts = []
    i=0
    for i,each in enumerate(poss_instances):
        inputs= each['input']
        outputs = each['output']
        input_text = each['input'][0]['text'].replace('<image0>',f'<image{i}>').replace('image 0',f'image {i}')
        input_image = inputs[1]['image'] 
        output_text = outputs[0][output_answer] 
        output_text = output_text.replace('<image0>',f'<image{i}>').replace('image 0',f'image {i}')
        texts.append("User: "+input_text+"\nBot: "+output_text+"\n")
        images.append(input_image)
    text = text.replace('<image0>',f'<image{i+1}>').replace('image 0',f'image {i+1}') if i>0 else text
    texts.append("User: "+text)
    out_text = out_text.replace('<image0>',f'<image{i+1}>').replace('image 0',f'image {i+1}') if i>0 else out_text 
    images.append(image)
    return "\n".join(texts) , images , out_text


def getsample_qa(path,types,split_num = -1,sample_icl=5,nature = False,dataset_name ='vqa',store_path=""):
    with open(path,'r') as f:
        js = json.load(f)
    result =[]
    if split_num == -1:
        instances = js['instances']
    else:
        instances = sample(js['instances'],split_num)
    for each in tqdm(instances):
        inputs= each['input']
        outputs = each['output']
        input_text = each['input'][0]['text'] 
        if len(each['input']) ==3:
            input_image = [each['input'][1]['image']] +each['input'][2]['bbox_list']
        else:
            input_image = each['input'][1]['image'] 
        if len(input_image) > 10 and isinstance(input_image,list):
            continue
        output_text = "Bot: "+each['output'][0]['answer'] 
        if output_text =="":
            continue
        output_image = ""
        input_text,input_image,output_text = qa_form_fewshot(instances,input_text,input_image,sample_icl,nature,output_text)
        js_dict ={'input_text':input_text,'input_image':input_image,'output_text':output_text,'output_image':output_image}
        result.append(js_dict)
    if nature:
        if split_num == -1:    
            filename = f'{dataset_name}-promot-1-{types}_nature.json'
        else:
            filename = f'{dataset_name}-promot-1-{types}_sample_{split_num}_nature.json'
    else:
        if split_num == -1:    
            filename = f'{dataset_name}-promot-1-{types}.json'
        else:
            filename = f'{dataset_name}-promot-1-{types}_sample_{split_num}.json'
    with open(os.path.join(store_path,filename),'w') as f:
        for dic in result:
            f.write(json.dumps(dic, ensure_ascii=False)+"\n") 

def nlvr2_form_fewshot(instances,text,image,sample_num,nature,output_answer='answer'):
    poss_instances= sample(instances,random.randint(0,sample_num))
    images=[]
    texts = []
    i=0
    for each in poss_instances:
        inputs= each['input']
        outputs = each['output']
        input_text = each['input'][0]['text'].replace('<image0>',f'<image{i}>').replace('image 0',f'image {i}').replace('<image1>',f'<image{i+1}>').replace('image 1',f'image {i+1}')
        input_image = inputs[1]['image'].split("###")
        output_text = outputs[0][output_answer] 
        texts.append("User: "+input_text+"\nBot: "+output_text+"\n")
        images.extend(input_image)
        i+=2
    text = text.replace('<image0>',f'<image{i}>').replace('image 0',f'image {i}').replace('<image1>',f'<image{i+1}>').replace('image 1',f'image {i+1}') if i>0 else text
    texts.append("User: "+text)
    images.extend(image)
    return "\n".join(texts) , images


def getsample_nlvr2(path,types,split_num = -1,sample_icl=5,nature = False,dataset_name ='vqa',store_path=""):
    with open(path,'r') as f:
        js = json.load(f)
    result =[]
    if split_num == -1:
        instances = js['instances']
    else:
        instances = sample(js['instances'],split_num)
    for each in tqdm(instances):
        inputs= each['input']
        outputs = each['output']
        input_text = each['input'][0]['text'] 
        input_image = each['input'][1]['image'].split("###") 
        if len(input_image) > 10 and isinstance(input_image,list):
            continue
        output_text = "Bot: "+each['output'][0]['answer'] 
        if output_text =="":
            continue
        output_image = ""
        input_text,input_image = nlvr2_form_fewshot(instances,input_text,input_image,sample_icl,nature)
        js_dict ={'input_text':input_text,'input_image':input_image,'output_text':output_text,'output_image':output_image}
        result.append(js_dict)
    if nature:
        if split_num == -1:    
            filename = f'{dataset_name}-promot-1-{types}_nature.json'
        else:
            filename = f'{dataset_name}-promot-1-{types}_sample_{split_num}_nature.json'
    else:
        if split_num == -1:    
            filename = f'{dataset_name}-promot-1-{types}.json'
        else:
            filename = f'{dataset_name}-promot-1-{types}_sample_{split_num}.json'
    with open(os.path.join(store_path,filename),'w') as f:
        for dic in result:
            f.write(json.dumps(dic, ensure_ascii=False)+"\n") 

def getsample_funqa(path,types,split_num = -1,dataset_name="ivqa",store_path =''):
    with open(path,'r') as f:
        js = json.load(f)
    result =[]
    for each in tqdm(js['instances']):
        inputs= each['input']
        outputs = each['output']
        input_text = "User: "+each['input'][0]['text'] 
        video = each['input'][1]['image'].replace('funqa',"funqa_cliped")
        input_image = [ video.replace(".mp4",f"_{i}.jpg") for i in range(12)]
        output_text = "Bot: "+each['output'][0]['answer'] 
        output_image = ""
        js_dict ={'input_text':input_text,'input_image':input_image,'output_text':output_text,'output_image':output_image}
        result.append(js_dict)
    if split_num == -1:
        with open(os.path.join(store_path,f'{dataset_name}-promot-1-{types}_nature.json'),'w') as f:
            for dic in result:
                f.write(json.dumps(dic, ensure_ascii=False)+"\n") 
    else:
        with open(os.path.join(store_path,f'{dataset_name}-promot-1-{types}_sample_{split_num}_nature.json'),'w') as f:
            for dic in sample(result,split_num):
                f.write(json.dumps(dic, ensure_ascii=False)+"\n")

def getsample_video(path,types,split_num = -1,dataset_name="funqa",store_path =''):
    with open(path,'r') as f:
        js = json.load(f)
    result =[]
    for each in tqdm(js['instances']):
        inputs= each['input']
        outputs = each['output']
        input_text = "User: "+each['input'][0]['text'] 
        input_image = each['input'][1]['image'] 
        output_text = "Bot: "+each['output'][0]['answer'] 
        output_image = ""
        js_dict ={'input_text':input_text,'input_image':input_image,'output_text':output_text,'output_image':output_image}
        result.append(js_dict)
    if split_num == -1:
        with open(os.path.join(store_path,f'{dataset_name}-promot-1-{types}_nature.json'),'w') as f:
            for dic in result:
                f.write(json.dumps(dic, ensure_ascii=False)+"\n") 
    else:
        with open(os.path.join(store_path,f'{dataset_name}-promot-1-{types}_sample_{split_num}_nature.json'),'w') as f:
            for dic in sample(result,split_num):
                f.write(json.dumps(dic, ensure_ascii=False)+"\n")


def caption_form_fewshot(instances,text,image,sample_num,nature,output_answer='answer'):
    poss_instances= sample(instances,random.randint(0,sample_num))
    images=[]
    texts = []
    i=0
    for i,each in enumerate(poss_instances):
        inputs= each['input']
        outputs = each['output']
        input_text = each['input'][0]['text'].replace('<image0>',f'<image{i}>').replace('image 0',f'image {i}')
        input_image = inputs[1]['image'] 
        output_text = outputs[0][output_answer].split("##")[0]
        texts.append(input_text+" "+output_text+"\n")
        images.append(input_image)
    text = text.replace('<image0>',f'<image{i+1}>').replace('image 0',f'image {i+1}') if i>0 else text
    texts.append(text)
    images.append(image)
    return "\n".join(texts) , images

def getsample_caption(path,types,split_num = -1,sample_icl=5,nature = False,dataset_name ='vqa',store_path=""):
    with open(path,'r') as f:
        js = json.load(f)
    result =[]
    if split_num == -1:
        instances = js['instances']
    else:
        instances = sample(js['instances'],split_num)
    for each in tqdm(instances):
        inputs= each['input']
        outputs = each['output']
        input_text = each['input'][0]['text'] 
        if len(each['input']) ==3:
            input_image = [each['input'][1]['image']] +each['input'][2]['bbox_list']
        else:
            input_image = each['input'][1]['image'] 
        if len(input_image) > 10 and isinstance(input_image,list):
            continue
        output_text = each['output'][0]['caption']
        if output_text =="":
            continue
        output_image = ""
        input_text,input_image = caption_form_fewshot(instances,input_text,input_image,sample_icl,nature,'caption')
        js_dict ={'input_text':input_text,'input_image':input_image,'output_text':output_text,'output_image':output_image}
        result.append(js_dict)
    if nature:
        if split_num == -1:    
            filename = f'{dataset_name}-promot-1-{types}_nature.json'
        else:
            filename = f'{dataset_name}-promot-1-{types}_sample_{split_num}_nature.json'
    else:
        if split_num == -1:    
            filename = f'{dataset_name}-promot-1-{types}.json'
        else:
            filename = f'{dataset_name}-promot-1-{types}_sample_{split_num}.json'
    with open(os.path.join(store_path,filename),'w') as f:
        for dic in result:
            f.write(json.dumps(dic, ensure_ascii=False)+"\n") 
def generate_jsonl_data_from_instances(store_path):
    all_dataset = ['llava', 'textvqa', 'diffusiondb','msrvttqa','msrvtt', 'wikiart','nocaps', 'miniimage','vqa', 'vcr', 'stvqa', 'okvqa', 'nlvr2' ,'gqa', 'refcoco', 'coco' ,'flickr','vizwiz_caption','textcaps','ln_coco','funqa']
    # qa_dataset = ['vqa',  'stvqa', 'okvqa' ,'gqa','textvqa','wikiart','iconqa']
    general_dataset = ['vqa',  'stvqa', 'okvqa' ,'gqa','textvqa','wikiart','iconqa','refcoco', 'coco' ,'flickr','diffusiondb','miniimage','nocaps','vizwiz_caption','textcaps','ln_coco']
    # "nlvr2 is special"
    # caption_dataset = [ 'refcoco', 'coco' ,'flickr','diffusiondb','miniimage','nocaps','vizwiz_caption','textcaps','ln_coco']
    video_dataset=['msrvttqa','msrvtt']


    if not os.path.exists(store_path):
        os.makedirs(store_path)
        
    # set the prompt path
    data ={'refcoco': {'train': f'{path_dir}tasks/task00001-phrase_grounding-refcoco-prompt-1-subset-train.json'},
    'wikiart': {'train': f'{path_dir}tasks/task00002-image_generation-wikiart-prompt-1-subset-train.json',
            'val': f'{path_dir}tasks/task00003-image_generation-wikiart-prompt-1-subset-val.json'},
 'nlvr2': {'train': f'{path_dir}tasks/task00004-visual_question_answering-nlvr2-prompt-1-subset-train.json',
            'val': f'{path_dir}tasks/task00005-visual_question_answering-nlvr2-prompt-1-subset-val.json',
            'test': f'{path_dir}tasks/task00006-visual_question_answering-nlvr2-prompt-1-subset-test.json'},
 'textvqa': {'train': f'{path_dir}tasks/task00007-visual_question_answering-textvqa-prompt-1-subset-train.json'},
 'vqa': {'train': f'{path_dir}tasks/task00008-visual_question_answering-vqa-prompt-1-subset-train.json',
        'val': f'{path_dir}tasks/task00009-visual_question_answering-vqa-prompt-1-subset-val.json',
        'test': f'{path_dir}tasks/task00010-visual_question_answering-vqa-prompt-1-subset-test.json'},
 'gqa': {'train': f'{path_dir}tasks/task00011-visual_question_answering-gqa-prompt-1-subset-train.json',
        'val': f'{path_dir}tasks/task00012-visual_question_answering-gqa-prompt-1-subset-val.json',
        'test': f'{path_dir}tasks/task00013-visual_question_answering-gqa-prompt-1-subset-test.json'},
 'iconqa': {'train': f'{path_dir}tasks/task00014-visual_question_answering-iconqa-prompt-1-subset-train.json',
        'val': f'{path_dir}tasks/task00015-visual_question_answering-iconqa-prompt-1-subset-val.json',
        'test': f'{path_dir}tasks/task00016-visual_question_answering-iconqa-prompt-1-subset-test.json'},
 'stvqa': {'train': f'{path_dir}tasks/task00017-visual_question_answering-stvqa-prompt-1-subset-train.json'},
 'llava': {'train': f'{path_dir}tasks/task00018-visual_dialog-llava-prompt-1-subset-train.json'},
 'msrvtt': {'train': f'{path_dir}tasks/task00019-video_question_answering-msrvtt-prompt-1-subset-train.json',
        'val': f'{path_dir}tasks/task00020-video_question_answering-msrvtt-prompt-1-subset-val.json',
        'test': f'{path_dir}tasks/task00021-video_question_answering-msrvtt-prompt-1-subset-test.json'},
 'flickr': {'train': f'{path_dir}tasks/task00022-image_captioning-flickr-prompt-1-subset-train.json',
        'val': f'{path_dir}tasks/task00023-image_captioning-flickr-prompt-1-subset-val.json',
        'test': f'{path_dir}tasks/task00024-image_captioning-flickr-prompt-1-subset-test.json'},
 'vizwiz_caption': {'train': f'{path_dir}tasks/task00025-image_captioning-vizwiz_caption-prompt-1-subset-train.json',
        'val': f'{path_dir}tasks/task00026-image_captioning-vizwiz_caption-prompt-1-subset-val.json'},
 'nocaps': {'val': f'{path_dir}tasks/task00027-image_captioning-nocaps-prompt-1-subset-train.json'},
 'coco': {'train': f'{path_dir}tasks/task00028-image_captioning-coco-prompt-1-subset-train.json',
        'val': f'{path_dir}tasks/task00029-image_captioning-coco-prompt-1-subset-val.json'},
 'okvqa': {'train': f'{path_dir}tasks/task00030-visual_question_answering-okvqa-prompt-1-subset-train.json',
        'val': f'{path_dir}tasks/task00031-visual_question_answering-okvqa-prompt-1-subset-val.json'},
 'msrvttqa': {'train': f'{path_dir}tasks/task00032-video_question_answering-msrvttqa-prompt-1-subset-train.json',
        'val': f'{path_dir}tasks/task00033-video_question_answering-msrvttqa-prompt-1-subset-val.json',
        'test': f'{path_dir}tasks/task00034-video_question_answering-msrvttqa-prompt-1-subset-test.json'},
 'textcaps': {'train': f'{path_dir}tasks/task00035-image_captioning-textcaps-prompt-1-subset-train.json',
        'val': f'{path_dir}tasks/task00036-image_captioning-textcaps-prompt-1-subset-val.json'},
 'ln_coco': {'train': f'{path_dir}tasks/task00037-image_captioning-ln_coco-prompt-1-subset-train.json',
        'val': f'{path_dir}tasks/task00038-image_captioning-ln_coco-prompt-1-subset-val.json'},
 'miniimage': {'train': f'{path_dir}tasks/task00039-image_classification-miniimage-prompt-1-subset-train.json',
        'val': f'{path_dir}tasks/task00040-image_classification-miniimage-prompt-1-subset-val.json',
        'test': f'{path_dir}tasks/task00041-image_classification-miniimage-prompt-1-subset-test.json'},
 'diffusiondb': {'train': f'{path_dir}tasks/task00042-image_captioning-diffusiondb-prompt-1-subset-train.json'},
 'vcr': {'train': f'{path_dir}tasks/task00045-visual_question_answering-vcr-prompt-1-subset-train.json',
        'val': f'{path_dir}tasks/task00046-visual_question_answering-vcr-prompt-1-subset-val.json'},

 'funqa': {'train': f'{path_dir}tasks/task00043-video_question_answering-funqa-prompt-1-subset-train.json',
        'val': f'{path_dir}tasks/task00044-video_question_answering-funqa-prompt-1-subset-val.json'}
  }

    for datasetname in general_dataset:
        print(datasetname)
        files = data[datasetname]
        for each in files:
            getsample_qa(files[each],each,nature=True,sample_icl=IN_CONTEXT_SAMPLE_NUM,dataset_name=datasetname,store_path = store_path)

    print('vcr')
    files = data['vcr']
    for each in files:
        getsample_vcr(files[each],each,store_path = store_path)

    files=data['funqa']
    print('funqa')
    for each in files:
        getsample_funqa(files[each],each,dataset_name='funqa',store_path = store_path)
    
    files = data['llava']
    for each in files:
        print('llava')
        getsample_llava(files[each],each,store_path = store_path)
    for datasetname in video_dataset:
        print(datasetname)
        files = data[datasetname]
        for each in files:
            getsample_video(files[each],each,dataset_name=datasetname,store_path = store_path)
    files = data['nlvr2']
    for each in files:
        print('nlvr2')
        getsample_nlvr2(files[each],each,nature=True,sample_icl=4,dataset_name="nlvr2",store_path = store_path)


def get_json_file(file_path):
    js =[]
    with open(file_path,'r')as f:
        for line in f.readlines():
            js.append(json.loads(line))
    return js
def generate_new_json(data_json,data_size,file_name):
    json_train=[]
    json_test=[]
    json_val=[]
    if not os.path.exists(f'{path_dir}{save_dir_name}'):
        os.makedirs(f'{path_dir}{save_dir_name}')
    if not os.path.exists(f'{path_dir}{save_dir_name}/train'):
        os.makedirs(f'{path_dir}{save_dir_name}/train')
    if not os.path.exists(f'{path_dir}{save_dir_name}/val'):
        os.makedirs(f'{path_dir}{save_dir_name}/val')
    if not os.path.exists(f'{path_dir}{save_dir_name}/test'):
        os.makedirs(f'{path_dir}{save_dir_name}/test')
    for each in tqdm(data_size):
        if f'{each}_train' in data_json:
            js_train = sample(get_json_file(data_json[f'{each}_train']),data_size[each]['train']) if data_size[each]['train'] != -1 else get_json_file(data_json[f'{each}_train'])
            if data_size[each]['train'] !=0:
                sample_num = data_size[each]['train']
                with open(f'{path_dir}{save_dir_name}/train/{file_name}-{each}-sample_{sample_num}-train.json','w') as f:
                    for dic in js_train:
                        f.write(json.dumps(dic, ensure_ascii=False)+"\n") 
        else:
            js_train =[]
        if f'{each}_test' in data_json:
        
            js_test = sample(get_json_file(data_json[f'{each}_test']),data_size[each]['test']) if data_size[each]['test'] != -1 else get_json_file(data_json[f'{each}_test'])
            if data_size[each]['test'] !=0:
                sample_num = data_size[each]['test']
                with open(f'{path_dir}{save_dir_name}/test/{file_name}-{each}-sample_{sample_num}-test.json','w') as f:
                    for dic in js_test:
                        f.write(json.dumps(dic, ensure_ascii=False)+"\n") 
        else:
            js_test =[]
            
        if f'{each}_val' in data_json:
            js_val = sample(get_json_file(data_json[f'{each}_val']),data_size[each]['val']) if data_size[each]['val'] != -1 else get_json_file(data_json[f'{each}_val'])
            if data_size[each]['val'] !=0:
                sample_num = data_size[each]['val']
                with open(f'{path_dir}{save_dir_name}/val/{file_name}-{each}-sample_{sample_num}-val.json','w') as f:
                    for dic in js_val:
                        f.write(json.dumps(dic, ensure_ascii=False)+"\n") 
        else:
            js_val =[]
        json_train.extend(js_train)
        json_test.extend(js_test)
        json_val.extend(js_val)
        # random.shuffle(json_train)
        # random.shuffle(json_test)
        # random.shuffle(json_val)

def get_json_file(file_path):
    js =[]
    with open(file_path,'r')as f:
        for line in f.readlines():
            js.append(json.loads(line))
    return js
def save_pickle_img(path,file):
    with open(join(path_dir,path),'ab') as f:
        pk.dump(file,f)

def extract_frames(video_path, num_frames):
    clip = VideoFileClip(join(path_dir,video_path))
    duration = clip.duration
    frame_times = [duration * i / (num_frames + 1) for i in range(1, num_frames + 1)]
    
    frames = []
    for t in frame_times:
        frame = clip.get_frame(t)
        image = Image.fromarray(frame)
        frames.append(image)
    
    clip.close()
    
    return frames
# def read_image(postfix,img_path):
#     if postfix == 'png':
#         image = Image.open(join(path_dir,img_path))
#     elif postfix == 'h5':
#         image = h5py.File(join(path_dir,img_path), 'r')
#     else:
#         image = Image.open(join(path_dir, img_path))
#     return image

def read_image(img_path):
    image = Image.open(join(path_dir,img_path))
    return image



import io
def pil_image_to_base64(image):
    # Convert the image to bytes
    image_byte_array = io.BytesIO()
    format = image.format if image.format is not None else 'png'
    image.save(image_byte_array, format=format)
    image_bytes = image_byte_array.getvalue()

    # Encode the image bytes to base64
    base64_encoded = base64.b64encode(image_bytes).decode('utf-8')

    return base64_encoded

def process_image(image_list):
    # pool = Pool(processes=NUM_PROCESS)
    # images = pool.map(self.read_image, image_list)
    # pool.close()
    # pool.join()
    images = [read_image(img) for img in image_list ]
    return images
    
def preprocess_function(input_text,input_image,output_text,dataset_name=None): # only process images
    result = {"input_text": "", "input_image": [], "output_text": "", "output_image": ""}
    flag = isinstance(input_image,list)
    result["input_image"] = []
    result["input_text"] = input_text
    result["output_text"] = output_text
    if flag:
        input_image = [img.replace('_/',"_") for img in input_image]
        images = process_image(input_image)
        result["input_image"].append(processor(images = images,return_tensors="pt")['pixel_values'])
    else:
        img = read_image(input_image.replace('_/',"_"))
        result["input_image"].append(processor(images = [img ],return_tensors="pt")['pixel_values'])

    return result


def preprocess_function_video(input_text,input_image,output_text,dataset_name=None):
    result = {"input_text": "", "input_image": [], "output_text": "", "output_image": ""}
    flag = isinstance(input_image,list)
    result["input_image"] = []
    result["input_text"] = input_text
    result["output_text"] = output_text
    postfix = input_image[1:].split('.')[-1]
    img_path = input_image[1:] if input_image[0] == '.' and input_image[1] !='/'  else input_image
    img_path = img_path.replace('_/',"_")
    if postfix =='mp4':
        images = extract_frames(img_path,NUM_FRAMES)

        video_name = img_path.split('/')[-1].split('.')[0]

        for idx, img in enumerate(images):
            clip_imgs = video_name+f'_clip{idx}.jpg'
            img_path = f"{path_dir}data/video_cliped/{clip_imgs}"
            img.save(img_path)
            result["input_image"].append(f"./data/video_cliped/{clip_imgs}")

    return result

def concat_text_input_output( input_ids, input_atts, output_ids, output_atts):
    input_part_targets_len = []
    llm_tokens = {"input_ids": [], "attention_mask": []}
    this_input_ones = sum(input_atts)
    input_part_targets_len.append(this_input_ones)
    llm_tokens['input_ids'].append(
        np.concatenate([
            input_ids[:this_input_ones],
            output_ids[1:],
            input_ids[this_input_ones:]
        ])
    )
    llm_tokens['attention_mask'].append(
        np.concatenate([
            input_atts[:this_input_ones],
            output_atts[1:],
            input_atts[this_input_ones:]
        ])
    )
    llm_tokens['input_ids'] = np.stack(llm_tokens['input_ids'])
    llm_tokens['attention_mask'] = np.stack(llm_tokens['attention_mask'])
    return llm_tokens, input_part_targets_len



def preprocess_function_batched(result,input_text,output_text): # preprocess the text tokenization
    if 'vicuna' in model_type:
        processor.tokenizer.padding_side = "right"
        processor.tokenizer.truncation_side = 'left'
        replace_token = "".join(32*['<visual_embedding>'])
        
        input_text = input_text.replace('图',replace_token)
        re = processor.tokenizer(
            input_text,
            padding="longest",
            truncation=True,
            max_length=max_seq_length,
        )
        processor.tokenizer.truncation_side = 'right'
        out = processor.tokenizer(
            output_text,
            padding="longest",
            truncation=True,
            max_length=256)
        re, input_part_targets_len = concat_text_input_output(
        re['input_ids'],
        re['attention_mask'],
        out['input_ids'],
        out['attention_mask'],
        )
        re['input_ids'] = np.array(re['input_ids'],dtype=np.int32)
        re['attention_mask'] = np.array(re['attention_mask'],dtype=np.bool_)

        # do not apply loss to the padding
        targets = copy.deepcopy(re['input_ids'])
        targets[targets == processor.tokenizer.pad_token_id] = -100


        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        m= {
            'input_ids': re['input_ids'],
            'attention_mask': re['attention_mask'],
            'label': targets,
        }
        result.update(m)

    else:
        re= processor.tokenizer(input_text, padding='max_length', max_length=max_seq_length, truncation=True)
        re['input_ids'] = np.array(re['input_ids'],dtype=np.int32)
        re['attention_mask'] = np.array(re['attention_mask'],dtype=np.bool_)
        # result['label'] = np.array(processor.tokenizer(output_text, padding='max_length', max_length=32, truncation=True)["input_ids"],dtype=np.int32)
        out = processor.tokenizer(output_text, padding='max_length', max_length=128, truncation=True)
        result['label'] = np.array(out["input_ids"],dtype=np.int32)
        result['label_attention_mask'] = np.array(out["attention_mask"],dtype=np.bool_)
        result.update(re)
    return result

def process_raw_datajson_to_pickle(json_data,types):
    json_data = get_json_file(json_data)
    with ThreadPoolExecutor(max_workers=10) as executor:
        for each in tqdm(json_data):
            input_text = each['input_text']
            input_imgs = each['input_image']
            output_text = each['output_text']
            temp = preprocess_function(input_text,input_imgs,output_text)
            temp = preprocess_function_batched(temp,input_text,output_text)
            executor.submit(save_pickle_img, f"mmicl-prompt-{types}.pkl",temp)
def save_to_arrow(path,temp):
    with ArrowWriter(path=path,writer_batch_size=2500) as writer: 
        writer.write_batch(temp) 
        writer.finalize() 
def save_to_jsonl(path,temp):
    with jsonlines.open(path, 'w') as writer:
        for each in temp:
            writer.write(each)
import fastparquet
def save_to_parquet(path,temp):
    fastparquet.write(path, temp)



   
# def process_raw_datajson_to_arrow(json_data,file_name,types,sub_length = -1,big_file_name=None,dataset_name=None):
#     if big_file_name is None:
#         big_file_name = file_name
#     if dataset_name is not None:
#         big_file_name = join(big_file_name,dataset_name)
#     # if not os.path.exists(f'{big_file_name}/jsonl_data_{file_name}_{types}'):
#     #     os.makedirs(f'{big_file_name}/jsonl_data_{file_name}_{types}')
#     dir_path = f"{big_file_name}/arrow_data_{file_name}_{types}"
#     if not os.path.exists(dir_path):
#             os.makedirs(dir_path)
#     if sub_length>0:
#         json_data = json_data[:sub_length]
#     save_arrow_data={'input_text':[], 'input_image':[], 'output_text':[]}
#     # save_arrow_data=[]
#     index_arrow=0
#     threads = []
#     def process_each(each):
#         input_text = each['input_text']
#         input_imgs = each['input_image']
#         output_text = each['output_text']
#         try:
#             return preprocess_function(input_text, input_imgs, output_text, dataset_name)
#         except Exception as e:
#             print(e)
#             return None
        
#     with Pool(processes=NUM_PROCESS) as p:
#         for i, result in enumerate(tqdm(p.imap(process_each, json_data), total=len(json_data))):
#             if result is not None:
#                 for each in save_arrow_data:
#                     save_arrow_data[each].append(result[each])
#             # Save every 2500 results

#             if i % 5000 == 0 and i != 0:
#                 if sub_length>0:
#                     path = f"{big_file_name}/arrow_data_{file_name}_{types}/mmicl-temp-{types}-{index_arrow}-length{sub_length}.arrow"
#                 else:
#                     path = f"{big_file_name}/arrow_data_{file_name}_{types}/mmicl-temp-{types}-{index_arrow}.arrow"
#                 # save_to_arrow(path, save_arrow_data)
#                 t = threading.Thread(target=save_to_arrow, args=(path, save_arrow_data))
#                 threads.append(t)
#                 t.start()
#                 save_arrow_data={'input_text':[], 'input_image':[], 'output_text':[]}
#                 index_arrow += 1
#         for t in threads:
#             t.join()
#     if save_arrow_data:
#         if sub_length > 0:
#             path = f"{big_file_name}/arrow_data_{file_name}_{types}/mmicl-temp-{types}-{index_arrow}-length{sub_length}.arrow"
#         else:
#             path = f"{big_file_name}/arrow_data_{file_name}_{types}/mmicl-temp-{types}-{index_arrow}.arrow"
#         save_to_arrow(path, save_arrow_data)


def process_raw_datajson_to_arrow(json_data,file_name,types,sub_length = -1,big_file_name=None,dataset_name=None):
    if big_file_name is None:
        big_file_name = file_name
    if dataset_name is not None:
        big_file_name = join(big_file_name,dataset_name)
    # if not os.path.exists(f'{big_file_name}/jsonl_data_{file_name}_{types}'):
    #     os.makedirs(f'{big_file_name}/jsonl_data_{file_name}_{types}')
    dir_path = f"{big_file_name}/arrow_data_{file_name}_{types}"
    if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    if sub_length>0:
        json_data = json_data[:sub_length]
    save_arrow_data={'input_text':[], 'input_image':[], 'output_text':[]}
    # save_arrow_data=[]
    index_arrow=0
    threads = []
    if MULTIPLE_PROCESS_ACROSS_INSTANCE:
        def process_each(each):
            input_text = each['input_text']
            input_imgs = each['input_image']
            output_text = each['output_text']
            try:
                return preprocess_function(input_text, input_imgs, output_text, dataset_name)
            except Exception as e:
                print(e)
                return None
        with Pool(processes=NUM_PROCESS) as p:
            for i, result in enumerate(tqdm(p.imap(process_each, json_data), total=len(json_data))):
                if result is not None:
                    for each in save_arrow_data:
                        save_arrow_data[each].append(result[each])
                # Save every 1000 results

                if i % 1000 == 0 and i != 0:
                    if sub_length>0:
                        path = f"{big_file_name}/arrow_data_{file_name}_{types}/mmicl-temp-{types}-{index_arrow}-length{sub_length}.arrow"
                    else:
                        path = f"{big_file_name}/arrow_data_{file_name}_{types}/mmicl-temp-{types}-{index_arrow}.arrow"
                    # save_to_arrow(path, save_arrow_data)
                    t = threading.Thread(target=save_to_arrow, args=(path, save_arrow_data))
                    threads.append(t)
                    t.start()
                    save_arrow_data={'input_text':[], 'input_image':[], 'output_text':[]}
                    index_arrow += 1
            for t in threads:
                t.join()
        if save_arrow_data:
            if sub_length > 0:
                path = f"{big_file_name}/arrow_data_{file_name}_{types}/mmicl-temp-{types}-{index_arrow}-length{sub_length}.arrow"
            else:
                path = f"{big_file_name}/arrow_data_{file_name}_{types}/mmicl-temp-{types}-{index_arrow}.arrow"
            save_to_arrow(path, save_arrow_data)
    else:
        
        for i,each in enumerate(tqdm(json_data)):
            input_text = each['input_text']
            input_imgs = each['input_image']
            output_text = each['output_text']
            try:
                temp =  preprocess_function(input_text, input_imgs, output_text, dataset_name)
            except Exception as e:
                print(e)
                continue
            for each in save_arrow_data:
                save_arrow_data[each].append(temp[each])
            if i % 1000 == 0 and i != 0:
                if sub_length>0:
                    path = f"{big_file_name}/arrow_data_{file_name}_{types}/mmicl-temp-{types}-{index_arrow}-length{sub_length}.arrow"
                else:
                    path = f"{big_file_name}/arrow_data_{file_name}_{types}/mmicl-temp-{types}-{index_arrow}.arrow"
                # save_to_arrow(path, save_arrow_data)
                t = threading.Thread(target=save_to_arrow, args=(path, save_arrow_data))
                threads.append(t)
                t.start()
                save_arrow_data={'input_text':[], 'input_image':[], 'output_text':[]}
                index_arrow += 1
        for t in threads:
            t.join()
        if sub_length>0:
            path = f"{big_file_name}/arrow_data_{file_name}_{types}/mmicl-temp-{types}-{index_arrow}-length{sub_length}.arrow"
        else:
            path = f"{big_file_name}/arrow_data_{file_name}_{types}/mmicl-temp-{types}-{index_arrow}.arrow"
        save_to_arrow(path,save_arrow_data)
                                                                                                                                                                 
def to_pickle(file_name):
    train_js =f'{path_dir}prompt_data/{file_name}-train.json'
    test_js =f'{path_dir}prompt_data/{file_name}-test.json'
    val_js =f'{path_dir}prompt_data/{file_name}-val.json'
    print('start process training data')
    process_raw_datajson_to_pickle(train_js,'train')
    print('start process testing data')
    process_raw_datajson_to_pickle(test_js,'test')
    print('start process val data')
    process_raw_datajson_to_pickle(val_js,'val')

def to_arrow(file_name,length=-1,do_train = True,convert_file_name=None):
    if convert_file_name is None:
        convert_file_name = file_name
    train_js =f'{path_dir}prompt_data/{file_name}-train.json'
    test_js =f'{path_dir}prompt_data/{file_name}-test.json'
    val_js =f'{path_dir}prompt_data/{file_name}-val.json'
    if do_train:
        train_js = get_json_file(train_js)
        print('start process training data')
        process_raw_datajson_to_arrow(train_js,convert_file_name,'train',length)
    print('start process testing data')
    test_js = get_json_file(test_js)
    process_raw_datajson_to_arrow(test_js,convert_file_name,'test',length)
    print('start process val data')
    val_js = get_json_file(val_js)
    process_raw_datajson_to_arrow(val_js,convert_file_name,'val',length)

def zero_preprocess_json(json_file):
    re =[]
    for j in json_file:
        m={'output_image':""}
        if 'vcr' in j['input_image'][0]:
            m["input_text"] = "image 0 is <image0>图.\n"+j['input_text'].split('图.\n')[-1]
            m["input_image"] = [j["input_image"][0]]
            m['output_text'] = j["output_text"]
        else:
            image_id = len(j['input_text'].split('\n'))-1
            m["input_text"] = ("image 0 is <image0>图.\n"+j['input_text'].split('\n')[-1]).replace(f'image {image_id}','image 0').replace(f'<image{image_id}>','<image0>')
            m["input_image"] = [j["input_image"][-1]]
            m['output_text'] = j["output_text"]
        if 'vicuna'  in model_type:
            replace_token = "".join(32*['<visual_embedding>'])
            m['input_text'] = m['input_text'].replace('图',replace_token)
        re.append(m)
    return re
def zero_shot_to_arrow(file_name,length=-1,do_train = True,convert_file_name=None):
    if convert_file_name is None:
        convert_file_name = file_name
    train_js =f'{path_dir}prompt_data/{file_name}-train.json'
    test_js =f'{path_dir}prompt_data/{file_name}-test.json'
    val_js =f'{path_dir}prompt_data/{file_name}-val.json'
    if do_train: 
        train_js = get_json_file(train_js)[:length] if length>0 else get_json_file(train_js)
        zero_train_js = zero_preprocess_json(train_js)
        print('start process training data')
        process_raw_datajson_to_arrow(zero_train_js,convert_file_name,'train_zeroshot',length)
    print('start process testing data')
    test_js = get_json_file(test_js)[:length] if length>0 else get_json_file(test_js)
    zero_test_js = zero_preprocess_json(test_js)
    process_raw_datajson_to_arrow(zero_test_js,convert_file_name,'test_zeroshot',length)
    print('start process val data')
    val_js = get_json_file(val_js)[:length] if length>0 else get_json_file(val_js)
    zero_val_js = zero_preprocess_json(val_js)
    process_raw_datajson_to_arrow(zero_val_js,convert_file_name,'val_zeroshot',length)





# from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool

from functools import partial
def process_train_data(train,length,big_file_name):
    if '-1' not in train:
        dataset_name = train.split('/')[-1].split('-')[-3]
    else:
        dataset_name = train.split('/')[-1].split('-')[-4]

    train_json = get_json_file(train)
    print(f'start process testing data {dataset_name}')
    process_raw_datajson_to_arrow(train_json, convert_file_name, 'train', length, big_file_name, dataset_name)

def process_val_data(val,length,big_file_name):
    if '-1' not in val:
        dataset_name = val.split('/')[-1].split('-')[-3]
    else:
        dataset_name = val.split('/')[-1].split('-')[-4]

    val_json = get_json_file(val)
    print(f'start process testing data {dataset_name}')
    process_raw_datajson_to_arrow(val_json, convert_file_name, 'val', length, big_file_name, dataset_name)

def process_test_data(test,length,big_file_name):
    if '-1' not in test:
        dataset_name = test.split('/')[-1].split('-')[-3]
    else:
        dataset_name = test.split('/')[-1].split('-')[-4]

    test_json = get_json_file(test)
    print(f'start process testing data {dataset_name}')
    process_raw_datajson_to_arrow(test_json, convert_file_name, 'test', length, big_file_name, dataset_name)

def sample_dataset(data_js,process_dataset):
    
    d_j = []
    for each in data_js:
        for process in process_dataset:
            if process in each:
                d_j.append(each)
    return d_j

def to_arrowByDataset(file_name,length=-1,do_train = True,convert_file_name=None, process_dataset=[]):
    if convert_file_name is None:
        convert_file_name = file_name
    big_file_name = f"{save_dir_name}/{file_name}"
    train_js =glob(f'{path_dir}{save_dir_name}/train/*')
    test_js =glob(f'{path_dir}{save_dir_name}/test/*')
    val_js =glob(f'{path_dir}{save_dir_name}/val/*')

    # process_dataset = [] means preocess all datasets
    if len(process_dataset) >0 :
        train_js = sample_dataset(train_js,process_dataset)

        test_js = sample_dataset(test_js,process_dataset)
        
        val_js = sample_dataset(val_js,process_dataset)
    
    print("train dataset num :", len(train_js))
    print("test dataset num :", len(test_js))
    print("val dataset num :", len(val_js))
    if MULTIPLE_PROCESS_ACROSS_INSTANCE:
        # preprocess with multi-process across different instances of each dataset
        for train in train_js:
            if '-1' not in train:
                dataset_name = train.split('/')[-1].split('-')[-3]
            else:
                dataset_name = train.split('/')[-1].split('-')[-4]
            train_json = get_json_file(train)
            print(f'start process training data {dataset_name}')
            process_raw_datajson_to_arrow(train_json,convert_file_name,'train',length,big_file_name,dataset_name)
        for val in val_js:
            if '-1' not in val:
                dataset_name = val.split('/')[-1].split('-')[-3]
            else:
                dataset_name = val.split('/')[-1].split('-')[-4]
            val_json = get_json_file(val)
            print(f'start process val data {dataset_name}')
            process_raw_datajson_to_arrow(val_json,convert_file_name,'val',length,big_file_name,dataset_name)
        for test in test_js:

            if '-1' not in test:
                dataset_name = test.split('/')[-1].split('-')[-3]
            else:
                dataset_name = test.split('/')[-1].split('-')[-4]
            test_json = get_json_file(test)
            print(f'start process testing data {dataset_name}')
            process_raw_datajson_to_arrow(test_json,convert_file_name,'test',length,big_file_name,dataset_name)
        
    else:
        
        # preprocess with multi-process across different datasets rather than the instances of each dataset

        with Pool(processes=NUM_PROCESS) as pool:
            partial_process_train_data = partial(process_train_data, length=length, big_file_name=big_file_name)
            pool.map(partial_process_train_data, train_js)
        
            partial_process_test_data = partial(process_test_data, length=length, big_file_name=big_file_name)
            pool.map(partial_process_test_data, test_js)
        
            partial_process_val_data = partial(process_val_data, length=length, big_file_name=big_file_name)
            pool.map(partial_process_val_data, val_js)
 

def merge_jsonl_files(file_list):
    merged_data = []
    for file_name in file_list:
        with open(file_name, 'r') as f:
            for line in f:
                data = json.loads(line)
                merged_data.append(data)
    return merged_data

def save_jsonl_file(file_name, data):
    with open(file_name, 'w') as f:
        for line in data:
            json.dump(line, f)
            f.write('\n')
def merge_the_data_json(dir_name):
    test = merge_jsonl_files(glob(f"{dir_name}/test/*"))
    val = merge_jsonl_files(glob(f"{dir_name}/val/*"))
    train = merge_jsonl_files(glob(f"{dir_name}/train/*"))
    save_jsonl_file(f"{dir_name}/test.json",test)
    save_jsonl_file(f"{dir_name}/val.json",val)
    save_jsonl_file(f"{dir_name}/train.json",train)
 
# merge all dataset into train, val , text split, if needed
# merge_the_data_json(save_dir_name)   
    
if __name__ == '__main__': 
    model_type='vicuna'
    path_dir = '.'
    if 'vicuna' in model_type:
        model_name_or_path = 'Salesforce/instructblip-vicuna-7b'
    else:
        model_name_or_path = 'Salesforce/instructblip-flan-t5-xl'

    MULTIPLE_PROCESS_ACROSS_INSTANCE = True
    NUM_PROCESS=32
    IN_CONTEXT_SAMPLE_NUM=8
    NUM_FRAMES=8


    processor = InstructBlipProcessor.from_pretrained(
        model_name_or_path,
    )
    if 'vicuna' in model_type:
        sp = ["<visual_embedding>"]+[f"<image{i}>" for i in range(20)]
        processor.tokenizer.add_special_tokens({'additional_special_tokens':sp})
        max_seq_length = min(2048, processor.tokenizer.model_max_length)
    else:
        sp = ["图"]+[f"<image{i}>" for i in range(20)]
        sp = sp+processor.tokenizer.additional_special_tokens[len(sp):]
        processor.tokenizer.add_special_tokens({'additional_special_tokens':sp})
        max_seq_length = min(512, processor.tokenizer.model_max_length)


    file_name = 'mmicl-prompt-vicuna-multiinst_final_ver'
    convert_file_name = 'mmicl-prompt-vicuna-multiinst_final_ver'

    seed= 100
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    save_dir_name='MMICL_vicuna_10_10_max_8_figures_json'
    store_path=f'{path_dir}MMICL_vicuna_10_10_max_8_figures'

    # generate by the generated_jsonl_data_fuction
    data_json={
    'coco_train' : f"{store_path}/coco-promot-1-train_nature.json",
    'coco_val' :f"{store_path}/coco-promot-1-val_nature.json",
    # 'coco_test' : f"{store_path}/coco-promot-1-val_nature.json",

    'flickr_train' : f"{store_path}/flickr-promot-1-train_nature.json",
    'flickr_val' : f"{store_path}/flickr-promot-1-val_nature.json",
    'flickr_test' : f"{store_path}/flickr-promot-1-test_nature.json",

    'gqa_train' : f"{store_path}/gqa-promot-1-train_nature.json",
    'gqa_val' : f"{store_path}/gqa-promot-1-val_nature.json",
    'gqa_test' : f"{store_path}/gqa-promot-1-test_nature.json",



    'nlvr2_train' : f"{store_path}/nlvr2-promot-1-train_nature.json",
    'nlvr2_val' : f"{store_path}/nlvr2-promot-1-val_nature.json",
    'nlvr2_test' : f"{store_path}/nlvr2-promot-1-test_nature.json",

    'vqa_train' : f"{store_path}/vqa-promot-1-train_nature.json",
    'vqa_val' : f"{store_path}/vqa-promot-1-val_nature.json",
    'vqa_test' : f"{store_path}/vqa-promot-1-test_nature.json",


    'miniimage_train':f'{store_path}/miniimage-promot-1-train_nature.json' ,
    'miniimage_val':f'{store_path}/miniimage-promot-1-val_nature.json' ,
    'miniimage_test':f'{store_path}/miniimage-promot-1-test_nature.json' ,

    'wikiart_train':f'{store_path}/wikiart-promot-1-train_nature.json' ,
    'wikiart_val':f'{store_path}/wikiart-promot-1-val_nature.json' ,
    # 'wikiart_test':f'{store_path}/wikiart-promot-1-val_nature.json' ,


    'stvqa_train' : f"{store_path}/stvqa-promot-1-train_nature.json",
    # 'stvqa_val' : f"{store_path}/stvqa-promot-1-train_nature.json",
    # 'stvqa_test' : f"{store_path}/stvqa-promot-1-train_nature.json",

    'refcoco_train' : f"{store_path}/refcoco-promot-1-train_nature.json",
    # 'refcoco_val' : f"{store_path}/refcoco-promot-1-train_nature.json",
    # 'refcoco_test' : f"{store_path}/refcoco-promot-1-train_nature.json",

    'llava_train':f'{store_path}/llava-promot-1-train_nature.json' ,
    # 'llava_val':f'{store_path}/llava-promot-1-train_nature.json' ,
    # 'llava_test':f'{store_path}/llava-promot-1-train_nature.json' ,

    'textvqa_train':f'{store_path}/textvqa-promot-1-train_nature.json' ,
    # 'textvqa_val':f'{store_path}/textvqa-promot-1-train_nature.json' ,
    # 'textvqa_test':f'{store_path}/textvqa-promot-1-train_nature.json' ,

    'diffusiondb_train':f'{store_path}/diffusiondb-promot-1-train_nature.json' ,
    # 'diffusiondb_val':f'{store_path}/diffusiondb-promot-1-train_nature.json' ,
    # 'diffusiondb_test':f'{store_path}/diffusiondb-promot-1-train_nature.json' ,

    # 'nocaps_train':f'{store_path}/nocaps-promot-1-val_nature.json' ,
    'nocaps_val':f'{store_path}/nocaps-promot-1-val_nature.json' ,
    # 'nocaps_test':f'{store_path}/nocaps-promot-1-val_nature.json' ,

    'msrvtt_train':f'{store_path}/msrvtt-promot-1-train_nature.json',
    'msrvtt_val':f'{store_path}/msrvtt-promot-1-val_nature.json',
    'msrvtt_test':f'{store_path}/msrvtt-promot-1-test_nature.json',

    'msrvttqa_train':f'{store_path}/msrvttqa-promot-1-train_nature.json',
    'msrvttqa_val':f'{store_path}/msrvttqa-promot-1-val_nature.json',
    'msrvttqa_test':f'{store_path}/msrvttqa-promot-1-test_nature.json',

    'funqa_train':f'{store_path}/funqa-promot-1-train_nature.json',
    'funqa_val':f'{store_path}/funqa-promot-1-val_nature.json',

    'ln_coco_train':f'{store_path}/ln_coco-promot-1-train_nature.json',
    'ln_coco_val':f'{store_path}/ln_coco-promot-1-val_nature.json',
    # 'ln_coco_test':f'{store_path}/ln_coco-promot-1-test_nature.json',

    'gqa_train' : f"{store_path}/gqa-promot-1-train_nature.json",
    'gqa_val' : f"{store_path}/gqa-promot-1-val_nature.json",
    'gqa_test' : f"{store_path}/gqa-promot-1-test_nature.json",

    'okvqa_train' : f"{store_path}/okvqa-promot-1-train_nature.json",
    'okvqa_val' : f"{store_path}/okvqa-promot-1-val_nature.json",
    # 'okvqa_test' : f"{store_path}/okvqa-promot-1-val_nature.json",

    'vcr_train' : f"{store_path}/vcr-promot-1-train_nature.json",
    'vcr_val' : f"{store_path}/vcr-promot-1-val_nature.json",
    # 'vcr_test' : f"{store_path}/vcr-promot-1-val_nature.json",

    'vizwiz_caption_train':f'{store_path}/vizwiz_caption-promot-1-train_nature.json',
    'vizwiz_caption_val':f'{store_path}/vizwiz_caption-promot-1-val_nature.json',


    'iconqa_train' : f"{store_path}/iconqa-promot-1-train_nature.json",
    'iconqa_val' : f"{store_path}/iconqa-promot-1-val_nature.json",
    'iconqa_test' : f"{store_path}/iconqa-promot-1-test_nature.json",

    'textcaps_train':f'{store_path}/textcaps-promot-1-train_nature.json',
    'textcaps_val':f'{store_path}/textcaps-promot-1-val_nature.json',

    }

    # sampled data size
    data_size= {
        "vqa": {
            "train": 120000,
            "test": 1000,
            "val": 1000
        },
        "nlvr2": {
            "train": 50000,
            "test": 500,
            "val": 500
        },
        "coco": {
            "train": 100000,
            "test": 1000,
            "val": 1000
        },
        "flickr": {
            "train": 60000,
            "test": 1000,
            "val": 1000
        },

        "stvqa": {
            "train": 60000,
            "test": 0,
            "val": 0
        },
        "refcoco": {
            "train": 100000,
            "test": 0,
            "val": 0
        },

        "miniimage": {
            "train": 25000,
            "test": 500,
            "val": 500
        },
        "wikiart": {
            "train": 8000,
            "test": 500,
            "val": 500
        },
        "llava": {
            "train": 200000,
            "test": 0,
            "val": 0
        },
        "textvqa": {
            "train": 25000,
            "test": 0,
            "val": 0
        },
        "diffusiondb": {
            "train": 15000,
            "test": 0,
            "val": 0
        },
        "nocaps": {
            "train": 0,
            "test": 2500,
            "val": 2500
        },
        "msrvtt": {
            "train": 50000,
            "test": 1000,
            "val": 1000
        },
        "msrvttqa": {
            "train": 50000,
            "test": 1500,
            "val": 1500
        },
        "iconqa": {
            "train": 15000,
            "test": 1500,
            "val": 1500
        },
        "vizwiz_caption": {
            "train": 80000,
            "test": 1500,
            "val": 1500
        },
        "textcaps": {
            "train": 80000,
            "test": 1500,
            "val": 1500
        },
        "ln_coco":{
            "train": 120000,
            "test": 1500,
            "val": 1500
        },    
        "funqa":{
            "train": 150000,
            "test": 1500,
            "val": 1500
        },
        "vcr": {
            "train": 118000,
            "test": 2000,
            "val": 2000
        },
        "gqa": {
            "train": 120000,
            "test": 1000,
            "val": 1000
        },
        
        "okvqa": {
            "train": 8000,
            "test": 2000,
            "val": 2000
        },
    }



    # 1. generate data instances
    generate_jsonl_data_from_instances(store_path)
    # 2. sample the json data base on the data_size
    generate_new_json(data_json,data_size,file_name) 
    
    # 3. preprocess the jsonl data and stored into the arrow files 
    
    # 1.5M data may takes 3.5T space
    
    # preprocess_function is the function to preprocess the image data using the image_processor to encode the images as pixel numpy array
    
    # preprocess_function_batched is the function to preprocess the text data using the text tokenizer to encode the input prompts as tokens; 
    # For convenience,  we recommend to tokenize the input prompts during the traning process not in the data process script.
     
    to_arrowByDataset(file_name)
