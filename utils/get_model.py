import timm
from model.vitfs import *


def get_model(model_name, image_size):

    if any(
        kw in model_name
        for kw in [
            "efficientvit",
            "levit",
            "swin",
            "mobilevit",
            "nest",
            "pit",
        ]
    ) or model_name in [
        "vitfs_tiny_patch16_gap_reg4_dinov2_bn_init",
        "vitfs_tiny_patch16_gap_reg4_dinov2_init"
    ]:
        model = timm.create_model(
            model_name,
            pretrained=False,
            img_size=image_size,
        )
        print(f"Created model {model_name} with img_size={image_size}")
    else:
        model = timm.create_model(model_name, pretrained=False)
        print(f"Created model {model_name} with img_size=None")
    
    return model