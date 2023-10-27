
import copy as cp
from pathlib import Path

import common.lib_utils as libutils
import common.io_utils as ioutils
from metadata.template import task_tmpl, instance_tmpl, example_tmpl
from taskgen.utils import gather_templates, initialize_template
import h5py
from model.blip2 import Blip2Processor
from model.blip2 import Blip2Config
from PIL import Image
from tqdm import tqdm
def get_dataset(dataset_name, data_path, subset_size):
    data_fetcher_path = f"datasets_new.{dataset_name}.{dataset_name}.fetch_data"
    fetcher = libutils.get_obj_from_str(data_fetcher_path)
    if data_path:
        dataset_loader = fetcher(data_path, subset_size, split="train")
    else:
        dataset_loader = None
    return dataset_loader

def tokenize(processor,input_text,input_image,output_text,output_image = None):
    example={}
    if output_image is None:
        example['label'] = processor.tokenizer(output_text, padding='max_length', max_length=32, truncation=True)["input_ids"]
    else:
        postfix = output_image[1:].split('.')[-1]

        if postfix == 'png':
            output_image = Image.open(output_image)
        elif postfix == 'h5':
            output_image = h5py.File(output_image, 'r')
        else:
            output_image = Image.open(output_image)
        outputs = processor(images=output_image, text=output_text ,padding='max_length', max_length=32, truncation=True, return_tensors="pt")

        example['label'] = outputs['input_ids'].detach().cpu().numpy().tolist()
        example['pixel_values'] = outputs['pixel_values'].squeeze(0).detach().cpu().numpy().tolist()
    postfix = input_image[1:].split('.')[-1]
    if postfix == 'png':
        input_image = Image.open(input_image)
    elif postfix == 'h5':
        input_image = h5py.File(input_image, 'r')
    else:
        output_image = Image.open(output_image)
    inputs = processor(images=input_image, text=input_text,  padding='max_length', max_length=256, truncation=True, return_tensors="pt")
    example['pixel_values'] = inputs['pixel_values'].squeeze(0).detach().cpu().numpy().tolist()
    example['input_text'] = inputs['input_ids'].detach().cpu().numpy().tolist()
    example['attention_mask'] = inputs['attention_mask'].detach().cpu().numpy().tolist()

    return example

def pre_process(processor,instance,cfg):

    input_info = instance["input"]
    output_info = instance["output"]
    if isinstance(input_info[0][cfg.task.input[0]], list):
        examples =[]
        for i in range(len(input_info[0][cfg.task.input[0]])):
            input_text, input_image =input_info[0][cfg.task.input[0]][i],input_info[1][cfg.task.input[1]][i]
            if len(cfg.task.output) >1:
                output_text, output_image =output_info[0][cfg.task.output[0]][i],output_info[1][cfg.task.output[1]][i]
            else:
                output_text = output_info[0][cfg.task.output[0]][i]
                output_image = None
            
            example = tokenize(processor,input_text,input_image,output_text,output_image)
        examples.append(example)
    else:
        input_text, input_image =input_info[0][cfg.task.input[0]],input_info[1][cfg.task.input[1]]
        if len(cfg.task.output) >1:
            output_text, output_image =output_info[0][cfg.task.output[0]],output_info[1][cfg.task.output[1]]
        else:
            output_text = output_info[0][cfg.task.output[0]]
            output_image = None

        examples = tokenize(processor,input_text,input_image,output_text,output_image)
    return examples




def create_task(dataset, cfg,processor=None):
    """
    Apply all the taskgen on the given dataset and create tasks
    """

    prompt_templates = gather_templates(cfg.task.template_path)
    examples = ioutils.load_json(Path(cfg.task.example_path))
    prompts = []

    # Generate taskgen instances
    for subset_idx, subsets in enumerate(dataset):
        task = cp.deepcopy(task_tmpl)

        # Reference related
        task["name"] = [cfg.task.name]
        task["source"] = {x: [cfg.dataset.name] for x in cfg.dataset.modalities}
        task["url"] = ioutils.cfg2obj(cfg.dataset.url)
        task["contributors"] = ioutils.cfg2obj(cfg.contact.contributor)

        # Collect definition from all_task_list, left it as a separate file for the convenience of plugging crawled data.
        task_metadata = ioutils.load_json(cfg.metadata.path)
        task["definition"] = cfg.task.name if cfg.task.definition else task_metadata[cfg.task.name]

        # Positive and negative examples
        task["positive_examples"] = examples["positive"]
        task["negative_examples"] = examples["negative"]

        for subset in tqdm(subsets):
            for sample in subset:
                if 'image' in sample and isinstance(sample['image'],list):
                    idx= 0
                    for i in range(len(sample['image'])):
                        vocabulary = {key : val[i] for key, val in sample.items() if isinstance(val[i], str)}
                        all_prompt_initted = initialize_template(prompt_templates, vocabulary=vocabulary)
                        if len(all_prompt_initted) == 1:
                            prompt_initted = all_prompt_initted[0]
                            prompt_text = prompt_initted["prompt"]
                            if prompt_text:
                                sample['text'] = [prompt_text]*len(sample['image'])
                            instance = cp.deepcopy(instance_tmpl)
                            instance["id"] = ''
                            instance["input"] = [{key: sample[key][i]} for key in cfg.task.input]
                            instance["output"] = [{key: sample[key][i]} for key in cfg.task.output]
                            # instance["prompt"] = [prompt_text]
                            # instances = pre_process(processor,instance)
                            # for inst in instances:
                            #     task["instances"].append(inst)
                            task["instances"].append(instance)  
                        else:
                            prompt_input = all_prompt_initted[0]
                            prompt_output = all_prompt_initted[1]
                            prompt_text = prompt_input["prompt"]
                            if prompt_text:
                                sample['text'] = [prompt_text]*len(sample['image'])
                            prompt_answer = prompt_output["prompt"]
                            if prompt_answer:
                                sample['answer'] = [prompt_text]*len(sample['image'])     

                            instance = cp.deepcopy(instance_tmpl)
                            instance["id"] = ''
                            instance["input"] = [{key: sample[key][i]} for key in cfg.task.input]
                            instance["output"] = [{key: sample[key][i]} for key in cfg.task.output]
                            # instance["prompt"] = [prompt_text]
                            # instances = pre_process(processor,instance)
                            # for inst in instances:
                            #     task["instances"].append(inst)
                            task["instances"].append(instance)  
                elif isinstance(sample,list):
                    for s in sample:
                        vocabulary = {key : val for key, val in s.items() if isinstance(val, str)}
                        all_prompt_initted = initialize_template(prompt_templates, vocabulary=vocabulary)
                        if len(all_prompt_initted) == 1:
                            prompt_initted = all_prompt_initted[0]
                            prompt_text = prompt_initted["prompt"]
                            if prompt_text:
                                s['text'] = prompt_text
                            instance = cp.deepcopy(instance_tmpl)
                            instance["id"] = ''
                            instance["input"] = [{key: s[key]} for key in cfg.task.input]
                            instance["output"] = [{key: s[key]} for key in cfg.task.output]
                            # instance["prompt"] = [prompt_text]
                            # instance = pre_process(processor,instance,cfg)
                            task["instances"].append(instance)
                        else:
                            prompt_input = all_prompt_initted[0]
                            prompt_output = all_prompt_initted[1]
                            prompt_text = prompt_input["prompt"]
                            if prompt_text:
                                s['text'] = prompt_text
                            prompt_answer = prompt_output["prompt"]
                            if prompt_answer:
                                s['answer'] = prompt_answer                           
                            instance = cp.deepcopy(instance_tmpl)
                            instance["id"] = ''
                            instance["input"] = [{key: s[key]} for key in cfg.task.input]
                            instance["output"] = [{key: s[key]} for key in cfg.task.output]
                            # instance["prompt"] = [prompt_text]
                            # instance = pre_process(processor,instance,cfg)
                            task["instances"].append(instance)
                else:
                    vocabulary = {key : val for key, val in sample.items() if isinstance(val, str)}
                    all_prompt_initted = initialize_template(prompt_templates, vocabulary=vocabulary)
                    if len(all_prompt_initted) == 1:
                        prompt_initted = all_prompt_initted[0]
                        prompt_text = prompt_initted["prompt"]
                        if prompt_text:
                            sample['text'] = prompt_text
                        instance = cp.deepcopy(instance_tmpl)
                        instance["id"] = ''
                        instance["input"] = [{key: sample[key]} for key in cfg.task.input]
                        instance["output"] = [{key: sample[key]} for key in cfg.task.output]
                        # instance["prompt"] = [prompt_text]
                        # instance = pre_process(processor,instance,cfg)
                        task["instances"].append(instance)
                    else:                        
                        prompt_input = all_prompt_initted[0]
                        prompt_output = all_prompt_initted[1]
                        prompt_text = prompt_input["prompt"]
                        if prompt_text:
                            sample['text'] = prompt_text
                        prompt_answer = prompt_output["prompt"]
                        if prompt_answer:
                            sample['answer'] = prompt_answer
                        instance = cp.deepcopy(instance_tmpl)
                        instance["id"] = ''
                        instance["input"] = [{key: sample[key]} for key in cfg.task.input]
                        instance["output"] = [{key: sample[key]} for key in cfg.task.output]
                        # instance["prompt"] = [prompt_text]
                        # instance = pre_process(processor,instance,cfg)
                        task["instances"].append(instance)
                        
        prompts.append(task)
            # Comment this
        # break
    return prompts


def collect(cfg):
    """
        Collect all input, output, taskgen based on task over a specific dataset
    """
    dataset_name = cfg.dataset.name

    
    # processor = Blip2Processor.from_pretrained(
    #         "/home/haozhezhao/model/blip2-flan-t5-xl",
    #     )


    dataloader = get_dataset(dataset_name, cfg.dataset.data_path, cfg.dataset.subset_size)
    prompts = create_task(dataloader, cfg)

    return prompts