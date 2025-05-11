from dataset import EgoExoDataset
from tqdm import tqdm
from pathlib import Path
import time
import data
import torch
from models import imagebind_model
from models.imagebind_model import ModalityType

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
make_directory('embeddings/exo')

device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)

ngpus = torch.cuda.device_count()
if ngpus > 1:
    print("using",ngpus,"GPUs" )
    model = torch.nn.DataParallel(model)

model.eval()
model.to(device)

seen_ego_takes = [f.name.replace('.pt', '') for f in Path('embeddings/ego-original').iterdir() if f.is_file()]
seen_exo_takes = [f.name.replace('.pt', '') for f in Path('embeddings/exo').iterdir() if f.is_file()]
seen_takes = [f for f in seen_ego_takes if f in seen_exo_takes]
dataset = EgoExoDataset(TAKES_FILENAME, SPLITS_FILENAME, DATA_ROOT_DIR, split='test',skip_takes=seen_takes)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)



with torch.no_grad():
    for batch in tqdm(dataloader):
        ego_data, exo_data, take_uid = batch
        take_uid = take_uid[0]
        ego_data = ego_data[0]
        exo_data = exo_data[0]

        #print("time0",time.time())
        print(take_uid)

        ego_embeddings = model({ModalityType.VISION: ego_data})[ModalityType.VISION]
        torch.save(ego_embeddings,f'embeddings/ego-original/{take_uid}.pt')
        exo_embeddings = model({ModalityType.VISION: exo_data})[ModalityType.VISION]
        torch.save(exo_embeddings,f'embeddings/exo/{take_uid}.pt')

        #print("time1",time.time())
