import timm

def get_model(model_name, image_size=None):
    if image_size is not None:
        model = timm.create_model(model_name, pretrained=False, img_size=image_size)
    else:
        model = timm.create_model(model_name, pretrained=False)
    
    return model