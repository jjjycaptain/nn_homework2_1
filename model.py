from typing import Tuple, List

import torch
import torch.nn as nn
from torchvision import models

def get_model(model_name: str = "resnet50", num_classes: int = 101) -> Tuple[nn.Module, List[nn.Parameter], List[nn.Parameter]]:
    model_name = model_name.lower()

    if model_name == "resnet50":
        model = models.resnet50()
        state_dict = torch.load("./weights/resnet50-0676ba61.pth", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        head_params = list(model.fc.parameters())
    else:
        raise ValueError(f"Unsupported model '{model_name}'. Currently only 'resnet50' is implemented.")

    head_ids = set(id(p) for p in head_params)
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return model, backbone_params, head_params



if __name__ == "__main__":
    # Example usage
    model, backbone_params, head_params = get_model("resnet50", 101)
    print(f"Model: {model}")
    print(f"Number of backbone parameters: {len(backbone_params)}")
    print(f"Number of head parameters: {len(head_params)}")