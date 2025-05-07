from dataset import EgoExoDataset
from tqdm import tqdm
from pathlib import Path

from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

TAKES_FILENAME = 'SET THIS VALUE' # this should be the takes.json file path
SPLITS_FILENAME = 'SET THIS VALUE' # this should be the splits.json file path
DATA_ROOT_DIR = 'SET THIS VALUE' # this should be the path to directory containing takes.json


def make_directory(directory):
    path = Path(directory)
    try:
        path.mkdir()
        print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")

make_directory('embeddings')
make_directory('embeddings/ego-original')
make_directory('embeddings/exo')

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

dataset = EgoExoDataset(TAKES_FILENAME, SPLITS_FILENAME, DATA_ROOT_DIR, split='test')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

with torch.no_grad():
    for batch in tqdm(dataloader):
        ego_data, exo_data, take_uid = batch
        ego_data = torch.squeeze(ego_data)
        exo_data = torch.squeeze(exo_data)
        ego_embeddings = model({ModalityType.VISION: ego_data})[ModalityType.VISION]
        torch.save(f'embeddings/ego-original/{take_uid}.pt', ego_embeddings)
        exo_embeddings = model({ModalityType.VISION: exo_data})[ModalityType.VISION]
        torch.save(f'embeddings/exo/{take_uid}.pt', exo_embeddings)

