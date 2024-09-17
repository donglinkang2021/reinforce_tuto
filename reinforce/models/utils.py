import torch.nn as nn

def init_weights(model:nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)