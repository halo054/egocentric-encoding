
'''
group_1_labels = [f.name.replace('.pt', '') for f in pathlib.Path(group_1_dir).iterdir() if f.is_file()]
group_1_features = [torch.load(f"{group_1_dir}/{f}.pt", weights_only=False)[0] for f in group_1_labels]

group_1_features = torch.stack(group_1_features)
group_2_features = torch.stack(group_2_features)
'''


from dataset import *
from tqdm import tqdm
from pathlib import Path
import time
import copy
import data
import torch
import torch.nn as nn
from models import imagebind_model
from models.imagebind_model import ModalityType
from train_ego_naive import VisionEncoder

class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = imagebind_model.imagebind_huge(pretrained=True)
        self.ego_preprocessor = copy.deepcopy(model.modality_preprocessors["vision"])
        self.ego_trunk = copy.deepcopy(model.modality_trunks["vision"])
        self.ego_head= copy.deepcopy(model.modality_heads["vision"])
        self.ego_postprocessor= copy.deepcopy(model.modality_postprocessors["vision"])
    def forward(self, ego_video_data):
        B, S = ego_video_data.shape[:2]
        ego_video_data = ego_video_data.reshape(B * S, *ego_video_data.shape[2:])
        ego_video_data = self.ego_preprocessor(vision =  ego_video_data)
        trunk_inputs = ego_video_data["trunk"]
        head_inputs = ego_video_data["head"]
        ego_video_data = self.ego_trunk(**trunk_inputs)
        ego_video_data = self.ego_head(ego_video_data, **head_inputs)
        ego_video_data = self.ego_postprocessor(ego_video_data)
        ego_video_data = ego_video_data.reshape(B, S, -1)
        ego_video_data = ego_video_data.mean(dim=1)
        return ego_video_data

model = VisionEncoder()
model.requires_grad_(False)
model.ego_head.requires_grad_(True)


