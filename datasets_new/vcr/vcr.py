from torch.utils.data import DataLoader, Dataset

import common.io_utils as io
from os.path import join

class VCRDataset(Dataset):
    def __init__(self, root_path, split="train",subset_size=None):
        super().__init__()
        self.subset = split
        self.anno_dir = "./data/vcr"
        self.img_dir = join("./data/vcr", "vcr1images")
        self.subset_size = subset_size
        self._load_dataset()


    def _load_dataset(self):
        self.samples = io.load_jsonl(join(self.anno_dir ,f"cliped_vcr_{self.subset}.jsonl"))
        if self.subset_size is not None and self.subset_size<= len(self.samples) and self.subset_size !=-1:
            self.samples = self.samples[:self.subset_size]

                
    def __len__(self):
        return len(self.samples)

    def item_to_str(self, mixed_list,objects_list):
        transfer_list=[]
        for item in mixed_list:
            if isinstance(item,list):
                if len(item) == 1:
                    transfer_list.append(objects_list[item[0]])                
                elif len(item) == 2:
                    transfer_list.append(" ".join([objects_list[item[0]],'and',objects_list[item[1]]]))
                else:
                    for idx, each in enumerate(item):
                        # if idx == len(item)-2:
                        #     tran_unit = " ".join([objects_list[each],"and"])
                        if idx ==len(item)-1:
                            tran_unit = objects_list[each]
                        else:
                            tran_unit = "".join([objects_list[each],","])
                        transfer_list.append(tran_unit)
            else:
                transfer_list.append(str(item))
        return transfer_list

        
    def merge_sentence(self,words):
        sentence = ""
        for word in words:

            if word in [",", ".",":"]:
                sentence = sentence.rstrip() + word + " "
            else:
                sentence += word + " "

        sentence = sentence.strip()
        return sentence

    def __getitem__(self, i):
        sample = self.samples[i]
        objects_list = sample['objects']
        prompt_list = [f"image {i+1} is <image{i+1}>" for i in range(len(sample['objects']))]
        # objects_list = [f"<image{i+1}>" for i in range(len(sample['objects']))]
        objects_list = [f"image {i+1}" for i in range(len(sample['objects']))]
        # objects_list = [f"<image{i+1}>图</image{i+1}>" for i in range(len(sample['objects']))]
        prompt = "图.\n".join(["image 0 is <image0>"]+prompt_list)+"图.\n"
        choices = [ f"option {i}: "+ self.merge_sentence(self.item_to_str(choice,objects_list))  for i,choice in enumerate(sample["answer_choices"])]
        options = "\n".join(choices)
        question = self.merge_sentence(self.item_to_str(sample['question'],objects_list))
        answer_index = sample['answer_label']
        img_name = join(self.img_dir,sample["img_fn"])
        metadata = join(self.img_dir, sample['metadata_fn'])
        bbox_list = [join(self.anno_dir,box)  for box in sample['bbox_path']]
        
        rational = sample['rationale_choices'][sample['rationale_label']]

        reason =  self.merge_sentence(self.item_to_str(rational,objects_list))

        # image, question + 4 candidate choices, answer(choice id)
        return {
            "prompt": prompt, 
            "image": img_name,
            "question": question,
            "options":options,
            "metadata": str(metadata),
            "bbox_list": bbox_list,
            "answer_index": answer_index,
            "answer": choices[answer_index],
            "reason" : reason

        }

    @staticmethod
    def collate_fn(batch):
        return batch


def fetch_data(data_path, subset_size, split="train"):
    loaders=[]
    for split in ['train','val']:
        dataset = VCRDataset(root_path=data_path, split=split,subset_size=subset_size)
        loader = DataLoader(
            dataset=dataset,
            batch_size=256,
            shuffle=False,
            drop_last=True,
            collate_fn=VCRDataset.collate_fn,
            num_workers=0
        )
        loaders.append(loader)
    return loaders