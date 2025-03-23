import numpy as np
from sklearn import metrics


def get_vit_layer_id(var_name, num_max_layer, prefix=''):
    if var_name in (prefix + "cls_token", prefix + "mask_token", prefix + "pos_embed"):
        return 0
    elif var_name.startswith(prefix + "patch_embed"):
        return 0
    elif var_name.startswith(prefix + "embed_tokens") or var_name.startswith(prefix + "patchifier"):
        return 0
    elif var_name.startswith(prefix + "rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith(prefix + "layers"):
        names = var_name.split('.')
        anchor_ind = names.index('layers') # 'blocks' is an anchor
        block_id = int(names[anchor_ind + 1])
        return block_id + 1
    elif var_name.startswith(prefix + "blocks"):
        names = var_name.split('.')
        anchor_ind = names.index('blocks') # 'blocks' is an anchor
        block_id = int(names[anchor_ind + 1])
        return block_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        if layer_id is not None:
            return self.values[layer_id]
        else:
            return 1

    def get_layer_id(self, var_name, prefix=''):
        return get_vit_layer_id(var_name, len(self.values), prefix)

def get_parameter_groups(args, model, get_layer_id=None, get_layer_scale=None, verbose=False):
    weight_decay = args.weight_decay


    if hasattr(model, 'no_weight_decay'):
        skip_list = model.no_weight_decay()
    else:
        skip_list = {}
    print(skip_list)
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        # if name in ["module.vit.cls_token", "module.vit.pos_embed"]:
        if "model.embed_tokens" in name or "model.pos_embed" in name:
            group_name = "no_decay"
            this_weight_decay = 0.
        elif len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_layer_id is not None:
            layer_id = get_layer_id(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    if verbose:
        import json
        print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    else:
        print("Param groups information is omitted...")
    return list(parameter_group_vars.values())


def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def ROC(label, y_prob):
    fpr, tpr, thresholds = metrics.roc_curve(label, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return roc_auc, optimal_th




