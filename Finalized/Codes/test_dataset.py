from dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm

dataset = EgoExoDataset(
    '/fs/vulcan-datasets/Ego-Exo4D/takes.json',
    '/fs/vulcan-datasets/Ego-Exo4D/annotations/splits.json',
    '/fs/vulcan-datasets/Ego-Exo4D',
    device='cpu')

audio_dataset = AudioDataset(
    '/fs/vulcan-datasets/Ego-Exo4D/takes.json',
    '/fs/vulcan-datasets/Ego-Exo4D/annotations/splits.json',
    '/fs/vulcan-datasets/Ego-Exo4D',
    device='cpu')
dataloader = DataLoader(dataset, batch_size=1,num_workers = 2)
for element in tqdm(dataloader):
    a = audio_dataset[element[2][0]]