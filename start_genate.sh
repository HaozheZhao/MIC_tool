#!/bin/bash

template_paths=(
'prompts/phrase_grounding/refcoco-prompt-1/configs.yaml'
'prompts/image_generation/wikiart-prompt-1/configs.yaml'
'prompts/image_generation/diffusiondb-prompt-1/configs.yaml'
'prompts/visual_question_answering/vcr-prompt-1/configs.yaml'
'prompts/visual_question_answering/nlvr2-prompt-1/configs.yaml'
'prompts/visual_question_answering/textvqa-prompt-1/configs.yaml'
'prompts/visual_question_answering/vqa-prompt-1/configs.yaml'
'prompts/visual_question_answering/gqa-prompt-2/configs.yaml'
'prompts/visual_question_answering/iconqa-prompt-1/configs.yaml'
'prompts/visual_question_answering/okvqa-prompt-1/configs.yaml'
'prompts/visual_question_answering/stvqa-prompt-1/configs.yaml'
'prompts/visual_dialog/llava-prompt-1/configs.yaml'
'prompts/video_question_answering/msrvttqa-prompt-1/configs.yaml'
'prompts/video_question_answering/msrvtt-prompt-1/configs.yaml'
'prompts/image_classification/miniimage-prompt-1/configs.yaml'
'prompts/image_captioning/flickr-prompt-1/configs.yaml'
'prompts/image_captioning/vizwiz_caption-prompt-1/configs.yaml'
'prompts/image_captioning/textcaps-prompt-1/configs.yaml'
'prompts/image_captioning/nocaps-prompt-1/configs.yaml'
'prompts/image_captioning/ln_coco-prompt-2/configs.yaml'
'prompts/image_captioning/fun_qa-prompt-1/configs.yaml'
)

for template_path in "${template_paths[@]}"
do
    echo $template_path
    python main.py --cfg $template_path
done