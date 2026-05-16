import json

import torch

def setup_model_hier_labels(train_file, model, tokenizer):
    hier_labels = None
    for line in open(train_file):
        if hier_labels:
            for i, l in enumerate(json.loads(line)['tgt']):
                hier_labels[i] |=  set(l)
                print(f"hier_labels = {hier_labels}")
        else:
            hier_labels = [set(i) for i in json.loads(line)['tgt']]
            print(f"hier_labels = {hier_labels}")
    assert hier_labels is not None
    hier_labels = [tokenizer.convert_tokens_to_ids(list([j.lower() for j in i])) for i in hier_labels]
    print(f"[DEBUG] hier_labels = {hier_labels}")

    def to_multi_hot(label):
        _label = torch.zeros(model.config.vocab_size)
        for i in label:
            _label[i] = 1
        return _label.bool()

    model.hier_labels = [to_multi_hot(i) for i in hier_labels]
    model.soft_label_hier_real = True
