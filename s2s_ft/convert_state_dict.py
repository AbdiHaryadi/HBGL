import torch
import logging

logger = logging.getLogger(__name__)


def hf_roberta_to_hf_bert(state_dict):
    logger.info(" * Convert Huggingface RoBERTa format to Huggingface BERT format * ")

    new_state_dict = {}

    for key in state_dict:
        value = state_dict[key]
        if key == 'roberta.embeddings.position_embeddings.weight':
            value = value[2:]
        if key == 'roberta.embeddings.token_type_embeddings.weight':
            continue
        if key.startswith('roberta'):
            key = 'bert.' + key[8:]
        elif key.startswith('lm_head'):
            if 'layer_norm' in key or 'dense' in key:
                key = 'cls.predictions.transform.' + key[8:]
            else:
                key = 'cls.predictions.' + key[8:]
            key = key.replace('layer_norm', 'LayerNorm')

        new_state_dict[key] = value

    return new_state_dict


def hf_electra_to_hf_bert(state_dict):
    logger.info(" * Convert Huggingface ELECTRA format to Huggingface BERT format * ")

    new_state_dict = {}

    for key in state_dict:
        value = state_dict[key]
        if key.startswith('electra'):
            key = 'bert.' + key[8:]
        new_state_dict[key] = value

    return new_state_dict


def hf_bert_to_hf_bert(state_dict):
    # keep no change
    return state_dict


def unilm_to_hf_bert(state_dict):
    logger.info(" * Convert Fast QKV format to Huggingface BERT format * ")

    new_state_dict = {}

    for key in state_dict:
        value = state_dict[key]
        if key.endswith("attention.self.q_bias"):
            new_state_dict[key.replace("attention.self.q_bias", "attention.self.query.bias")] = value.view(-1)
        elif key.endswith("attention.self.v_bias"):
            new_state_dict[key.replace("attention.self.v_bias", "attention.self.value.bias")] = value.view(-1)
            new_state_dict[key.replace("attention.self.v_bias", "attention.self.key.bias")] = torch.zeros_like(value.view(-1))
        elif key.endswith("attention.self.qkv_linear.weight"):
            l, _ = value.size()
            assert l % 3 == 0
            l = l // 3
            q, k, v = torch.split(value, split_size_or_sections=[l, l, l], dim=0)
            new_state_dict[key.replace("attention.self.qkv_linear.weight", "attention.self.query.weight")] = q
            new_state_dict[key.replace("attention.self.qkv_linear.weight", "attention.self.key.weight")] = k
            new_state_dict[key.replace("attention.self.qkv_linear.weight", "attention.self.value.weight")] = v
        elif key == "bert.encoder.rel_pos_bias.weight":
            new_state_dict["bert.rel_pos_bias.weight"] = value
        else:
            new_state_dict[key] = value

    del state_dict

    return new_state_dict


state_dict_convert = {
    'bert': hf_bert_to_hf_bert,
    'unilm': unilm_to_hf_bert, 
    'minilm': hf_bert_to_hf_bert, 
    'roberta': hf_roberta_to_hf_bert,
    'xlm-roberta': hf_roberta_to_hf_bert,
    'electra': hf_electra_to_hf_bert,
}
