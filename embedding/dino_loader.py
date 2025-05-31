from transformers import AutoImageProcessor, AutoModel
import torchvision.models as models
import torch

target_features = {}

def hook_fn(module, input, output):
    target_features['layer4'] = output

def init_dino(model_name, device):
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name, output_attentions=True).to(device)
    model.eval()
    return processor, model

def init_target_model(device):
    model = models.resnet50(pretrained=True).to(device)
    model.eval()
    model.layer4.register_forward_hook(hook_fn)
    return model