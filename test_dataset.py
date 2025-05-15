from dataset import *
from torch.utils.data import DataLoader


dataset = EgoExoDataset(
    '/mnt/home/derek/cmsc/848M/ego-imagebind/egocentric-encoding/example_takes.json',
    '/mnt/home/derek/cmsc/848M/ego-imagebind/Ego-Exo4D_Example/annotations/splits.json',
    '/mnt/home/derek/cmsc/848M/ego-imagebind/Ego-Exo4D_Example/',
    device='cpu')

audio_dataset = AudioDataset(
    '/mnt/home/derek/cmsc/848M/ego-imagebind/egocentric-encoding/example_takes.json',
    '/mnt/home/derek/cmsc/848M/ego-imagebind/Ego-Exo4D_Example/annotations/splits.json',
    '/mnt/home/derek/cmsc/848M/ego-imagebind/Ego-Exo4D_Example/',
    device='cpu')
dataloader = DataLoader(dataset, batch_size=1)
for element in dataloader:
    print(audio_dataset[element[2][0]])