import torch

from .unetr import UNETR
from .base_model import BaseModel


class Model_Config():
    def __int__(self):
        pass


def build_model(args, roi_size):
    in_channels = args.in_channels
    n_class = args.out_channels

    model = UNETR(
        in_channels=in_channels,
        out_channels=n_class,
        img_size=roi_size,
        patch_size=[args.patch_size, args.patch_size, args.patch_size]
    )

    config = Model_Config()
    config.pos_type = args.pos_type
    config.img_size = roi_size
    config.patch_size = [args.patch_size, args.patch_size, args.patch_size]
    config.hidden_size = 768
    config.intermediate_size = 3072
    config.num_attention_heads = 12
    config.num_key_value_heads = 12
    config.num_hidden_layers = 12
    encoder = BaseModel(config)
    model_size = sum(t.numel() for t in encoder.parameters())
    print('model_size', model_size)

    if args.pretrain_path is not None:
        model_dict = torch.load(args.pretrain_path)["state_dict"]
        pretrained_state = {k.split('model.')[-1]: v for k, v in model_dict.items() if k.startswith('model')}
        encoder.load_state_dict(pretrained_state, strict=True)

    model.vit = encoder
    return model

    
