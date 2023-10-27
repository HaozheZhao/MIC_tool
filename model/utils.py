
from transformers import (
    BertModel,
    RobertaModel,
    AlbertModel,
    DebertaV2Model,
    XLNetModel,
    DebertaV2Model,
    AutoConfig
)

from model.blip2.modeling_blip_2 import Blip2ForConditionalGeneration

MODEL_CLASS = {
    "blip-2": Blip2ForConditionalGeneration
}


def get_model(model_args, config: AutoConfig, fix_bert: bool = False):

    model_class = MODEL_CLASS[config.model_type]
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=config
    )

    bert_param = 0
    if fix_bert:
        if config.model_type == "bert":
            for param in model.bert.parameters():
                param.requires_grad = False
            for _, param in model.bert.named_parameters():
                bert_param += param.numel()
        elif config.model_type == "roberta":
            for param in model.roberta.parameters():
                param.requires_grad = False
            for _, param in model.roberta.named_parameters():
                bert_param += param.numel()

    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
    total_param = all_param - bert_param

    for param in model.parameters():
        param.requires_grad = False
        model.query_tokens.requires_grad = True
    for param in model.qformer.parameters():
        param.requires_grad = True
    for param in model.language_projection.parameters():
        param.requires_grad = True

    print('***** total param is {} *****'.format(total_param))
    return model
