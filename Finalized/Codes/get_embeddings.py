from dataset import *
from tqdm import tqdm
from pathlib import Path
import time
import data
import torch
from models import imagebind_model
from models.imagebind_model import ModalityType
from train_ego_naive import VisionEncoder
import os
TAKES_FILENAME = '/fs/vulcan-datasets/Ego-Exo4D/takes.json' # this should be the takes.json file path
SPLITS_FILENAME = '/fs/vulcan-datasets/Ego-Exo4D/annotations/splits.json' # this should be the splits.json file path
DATA_ROOT_DIR = '/fs/vulcan-datasets/Ego-Exo4D' # this should be the path to directory containing takes.json


def make_directory(directory):
    path = Path(directory)
    try:
        path.mkdir()
        print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")

make_directory('embeddings')
make_directory('embeddings/ego-original')
make_directory('embeddings/ego-finetuned')
make_directory('embeddings/exo')
make_directory('embeddings/exo-training')
make_directory('embeddings/audio')
device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate model

'''
model = imagebind_model.imagebind_huge(pretrained=True)
ngpus = torch.cuda.device_count()
if ngpus > 1:
    print("using",ngpus,"GPUs" )
    model = torch.nn.DataParallel(model)

model.eval()
model.to(device)

'''
'''
seen_ego_takes = [f.name.replace('.pt', '') for f in Path('embeddings/ego-original').iterdir() if f.is_file()]
seen_exo_takes = [f.name.replace('.pt', '') for f in Path('embeddings/exo').iterdir() if f.is_file()]
seen_takes = [f for f in seen_ego_takes if f in seen_exo_takes]
'''
audio_files = [f.name.replace('.wav', '') for f in Path('/cmlscratch/xyu054/Imagebind/ImageBind-LoRA/audio/test').iterdir() if f.is_file()]
video_embedding_files = [f.name.replace('.pt', '') for f in Path('embeddings/exo').iterdir() if f.is_file()]

avoid_takes = [f for f in video_embedding_files if f not in audio_files]
dataset = EgoExoDataset(TAKES_FILENAME, SPLITS_FILENAME, DATA_ROOT_DIR, split='test',skip_takes=avoid_takes)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,num_workers=8)

'''
egoencoder = VisionEncoder()
ngpus = torch.cuda.device_count()
if ngpus > 1:
    print("using",ngpus,"GPUs" )
    egoencoder = torch.nn.DataParallel(egoencoder)

egoencoder.eval()
egoencoder.to(device)
'''

model = imagebind_model.imagebind_huge(pretrained=True)

ngpus = torch.cuda.device_count()
if ngpus > 1:
    print("using",ngpus,"GPUs" )
    model = torch.nn.DataParallel(model)

model.eval()
model.to(device)


'''
def load_pretrained_models(model,ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
os.environ["DECORD_EOF_RETRY_MAX"] = "40960"
ego_model = VisionEncoder()
ngpus = torch.cuda.device_count()
if ngpus > 1:
    print("using",ngpus,"GPUs" )
    ego_model = torch.nn.DataParallel(ego_model)
    load_pretrained_models(ego_model,"/cmlscratch/xyu054/Imagebind/ImageBind-LoRA/ckpt/checkpoint_epoch_2.pth")
ego_model.eval().requires_grad_(False)
ego_model.to('cuda')
'''

        

audio_dataset = AudioDataset(
    '/fs/vulcan-datasets/Ego-Exo4D/takes.json',
    '/fs/vulcan-datasets/Ego-Exo4D/annotations/splits.json',
    '/fs/vulcan-datasets/Ego-Exo4D',
    device='cpu')

with torch.no_grad():
    for element in tqdm(dataloader):
        ego_data, exo_data, take_uid = element
        audio = audio_dataset[element[2][0]]
        audio = audio.to(device)

        #ego_data, exo_data, take_uid = batch
        #print(ego_data.shape)
        #print(exo_data.shape)
        take_uid = take_uid[0]
        
        #ego_data = ego_data[0]
        #exo_data = exo_data[0]
        #ego_data = ego_data.to(device)
        audio_embeddings = model({ModalityType.AUDIO: audio})[ModalityType.AUDIO]

        torch.save(audio_embeddings,f'embeddings/audio/{take_uid}.pt')
        '''
        #print("time0",time.time())
        print(take_uid)

        ego_embeddings = model({ModalityType.VISION: ego_data})[ModalityType.VISION]
        #torch.save(ego_embeddings,f'embeddings/ego-original/{take_uid}.pt')
        exo_embeddings = model({ModalityType.VISION: exo_data})[ModalityType.VISION]
        #torch.save(exo_embeddings,f'embeddings/exo/{take_uid}.pt')
        '''

        #print("time1",time.time())