# Based on PyTorch Lightning Tutorial 13 -
# SSL : https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/13-contrastive-learning.html
# Modified by Fares Abawi (@fabawi).
import logging
import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import copy
from models import imagebind_model
from models import lora as LoRA
from models.imagebind_model import ModalityType, load_module, save_module
from dataset import *
import os.path as osp
import yaml
import json
import numpy as np
from tqdm import tqdm
import time
#import matplotlib.pyplot as plt

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

'''
class ImageBindTrain(L.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=1e-4, max_epochs=500, batch_size=32, num_workers=4, seed=42, 
                 self_contrast=False, temperature=0.07,  momentum_betas=(0.9, 0.95), 
                 lora=False, lora_rank=4, lora_checkpoint_dir="./.checkpoints/lora",
                 lora_layer_idxs=None, lora_modality_names=None,
                 linear_probing=False,ego=False
                 ):
        super().__init__()
        
        self.save_hyperparameters()

        # Load full pretrained ImageBind model
        self.model = imagebind_model.imagebind_huge(pretrained=True)

        if ego:
            for modality_preprocessor in self.model.modality_preprocessors.children():
                modality_preprocessor.requires_grad_(False)
            for modality_trunk in self.model.modality_trunks.children():
                modality_trunk.requires_grad_(False)
            for modality_postprocessor in self.model.modality_postprocessors.children():
                modality_postprocessor.requires_grad_(False)

            ego_encoder = copy.deepcopy(model.modality_trunks["vision"])
            ego_encoder.requires_grad = True
            

    def info_nce_loss(self, batch, mode="train"):
        data_a, class_a, data_b, class_b = batch

        # class_a is always "vision" according to ImageBind
        feats_a = [self.model({class_a[0]: data_a_i}) for data_a_i in data_a]
        feats_a_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_a], dim=0)
        # class_b could be any modality
        feats_b = [self.model({class_b[idx]: data_b_i}) for idx, data_b_i in enumerate(data_b)]
        feats_b_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_b], dim=0)

        
        return dual_nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")

    def on_validation_epoch_end(self):
        if self.hparams.lora:
            # Save LoRA checkpoint
            LoRA.save_lora_modality_trunks(self.model.modality_trunks, checkpoint_dir=self.hparams.lora_checkpoint_dir)
            # Save postprocessors & heads
            save_module(self.model.modality_postprocessors, module_name="postprocessors",
                        checkpoint_dir=self.hparams.lora_checkpoint_dir)
            save_module(self.model.modality_heads, module_name="heads",
                        checkpoint_dir=self.hparams.lora_checkpoint_dir)
        elif self.hparams.linear_probing:
            # Save postprocessors & heads
            save_module(self.model.modality_heads, module_name="heads",
                        checkpoint_dir=self.hparams.lora_checkpoint_dir)    
'''


def get_exo_embedding(exo_dir,batch_labels):
        batch_exo_features = [torch.load(f"{exo_dir}/{f}.pt", weights_only=False)[0] for f in batch_labels]
        batch_exo_features = torch.stack(batch_exo_features)
        return batch_exo_features



class EgoTrainer:
    """Training and evaluation framework for material classification
    
    This class handles:
    1. Single modality classification (Task 1)
    2. Multi-modal fusion (Task 2)
    3. Contrastive learning (Task 3)
    """
    def __init__(self,ckpt_path,batch_size = 64):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.exo_dir = "embeddings/exo-training/"
        self.setup_environment()
        self.build_dataloaders()
        self.build_model()
        self.setup_optimization()
        
    def setup_environment(self):
        """Setup random seeds and computing device"""
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        

    def build_dataloaders(self):
        """Initialize train/val/test dataloaders"""
        TAKES_FILENAME = '/fs/vulcan-datasets/Ego-Exo4D/takes.json' # this should be the takes.json file path
        SPLITS_FILENAME = '/fs/vulcan-datasets/Ego-Exo4D/annotations/splits.json' # this should be the splits.json file path
        DATA_ROOT_DIR = '/fs/vulcan-datasets/Ego-Exo4D' # this should be the path to directory containing takes.json
        dataset = EgoExoDataset(TAKES_FILENAME, SPLITS_FILENAME, DATA_ROOT_DIR, split='train',skip_takes=[])
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size,num_workers=4,shuffle=True)

    def build_model(self):
        """Initialize model based on task type"""
        self.ego_model = VisionEncoder()
        self.ego_model.requires_grad_(False)
        self.ego_model.ego_head.requires_grad_(True)

        ngpus = torch.cuda.device_count()
        if ngpus > 1:
            print("using",ngpus,"GPUs" )
            self.ego_model = torch.nn.DataParallel(self.ego_model)
            #self.load_pretrained_models(self.ckpt_path)
        self.ego_model.train()
        self.ego_model.to(self.device)

    def load_pretrained_models(self,ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        self.ego_model.load_state_dict(ckpt["model_state_dict"], strict=True)


    def setup_optimization(self):
        """Setup optimizer and learning rate scheduler"""
        self.optimizer = optim.AdamW(
                self.ego_model.parameters(),
                lr=1e-3,
                weight_decay=1e-2
            )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50, #self.args.epochs
            eta_min=1e-6
        )


    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.ego_model.train()
        running_loss = 0.0
        current_ego_embedding = []
        current_exo_embedding = []
        
        for batch in tqdm(self.dataloader):
            ego_data, _ , take_uid = batch
            print(take_uid)
            #print("ego_data shape",ego_data.shape)
            '''
            B, S = ego_data.shape[:2]
            ego_data = ego_data.reshape(
                    B * S, *ego_data.shape[2:]
                )
            '''
            ego_data = ego_data.to(self.device)
            # forward propagation
            self.optimizer.zero_grad()
            ego_embeddings = self.ego_model(ego_data)
            exo_embeddings = get_exo_embedding(self.exo_dir,take_uid)
            exo_embeddings = exo_embeddings.requires_grad_(False).to(self.device)
            current_ego_embedding.append(ego_embeddings)
            
            current_exo_embedding.append(exo_embeddings)
            print("len(current_ego_embedding)*self.batch_size =",len(current_ego_embedding)*self.batch_size)
            
            current_ego_embedding = torch.cat(current_ego_embedding)
            current_exo_embedding = torch.cat(current_exo_embedding)
            loss = self.info_nce_loss(current_ego_embedding,current_exo_embedding) \
            + self.info_nce_loss(current_exo_embedding,current_ego_embedding)
            loss = loss/2

            # backward propagation
            loss.backward()
            self.optimizer.step()
            print(loss.item())
            running_loss += loss.item()
            current_ego_embedding = []
            current_exo_embedding = []
            


        
        # calculate average loss
        epoch_loss = running_loss / len(self.dataloader)
        self.train_losses.append(epoch_loss)

        return epoch_loss


    def train(self):
        """Main training loop"""
        #self.train_start = time.time()
        self.train_losses = []
        for epoch in range(5):
            # Training
            #self.epoch_start = time.time() 
            #print(f"Start of epoch {epoch}:",self.epoch_start - self.train_start)
            train_loss = self.train_epoch(epoch)

            # Update learning rate
            self.scheduler.step()

            print(f"Epoch {epoch}")
            print(f"Train Loss: {train_loss:.4f}")
            print()
            self.save_checkpoint(epoch)
        '''  
        plt.plot([0,1,2,3,4],self.train_losses)
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")    
        plt.savefig("/cmlscratch/xyu054/Imagebind/ImageBind-LoRA/loss_visualization.png")
        '''
        with open("/cmlscratch/xyu054/Imagebind/ImageBind-LoRA/loss.txt", "w") as file:
            for item in self.train_losses:
                file.write(str(item) + "\n")
            file.write("\n\n\n\n\n")
        

    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        # Save checkpoint for current epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.ego_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }
        

        torch.save(
            checkpoint,
            osp.join("/cmlscratch/xyu054/Imagebind/ImageBind-LoRA/ckpt/", f'checkpoint_epoch_{epoch}.pth')
            )
        print(f'checkpoint_epoch_{epoch}.pth saved')
        '''
        if get_rank() == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.ego_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
        

            torch.save(
                checkpoint,
                osp.join("/cmlscratch/xyu054/Imagebind/ImageBind-LoRA/ckpt/", f'checkpoint_epoch_{epoch}.pth')
                )
                '''


    def info_nce_loss(self, features_1, features_2):
        # TODO: Implement InfoNCE loss
        #Inspired from https://github.com/arashkhoeini/infonce/blob/main/infonce/infonce.py
        self.temperature = 0.07
        similarity = torch.matmul(features_1,features_2.t()) / self.temperature

        labels = torch.arange(features_1.size(0))
        pos_mask_matrix = (labels.unsqueeze(1) == labels.t().unsqueeze(0)).float().to(self.device)
        neg_mask_matrix = 1-pos_mask_matrix

        pos_mask_add = neg_mask_matrix * (-1000)
        neg_mask_add = pos_mask_matrix * (-1000)

        loss = torch.logsumexp((similarity * pos_mask_matrix+pos_mask_add),-1) - torch.logsumexp(similarity ,-1)
        loss = -torch.mean(loss)
        #print(loss)
        return loss

def main():
    os.environ["DECORD_EOF_RETRY_MAX"] = "40960"
    exo_dir = "embeddings/exo-training/"
    # Initialize trainer
    trainer = EgoTrainer(ckpt_path = "/cmlscratch/xyu054/Imagebind/ImageBind-LoRA/ckpt/checkpoint_epoch_0.pth",batch_size = 128,)
    #trainer.save_checkpoint(0)
    # Run training or evaluation

    trainer.train()


if __name__ == '__main__':
    main()
