from torch.utils.data import Dataset
import json
import utils
from pathlib import Path
import subprocess
import ffmpeg
# aria01_214-1.mp4 for the centered egocentric


class EgoExoDataset(Dataset):
    # split can be one of 'train', 'test', 'val'
    def __init__(self, takes_filename, splits_filename, data_root_dir, split='test', device='cuda', skip_takes=[]):
        self.device = device
        self.data_root_dir = data_root_dir
        with open(splits_filename) as splits_file:
            splits_map = json.load(splits_file)
        split_set = set(splits_map["split_to_take_uids"][split])
        del splits_map
        self.entries = []
        with open(takes_filename) as takes_file:
            takes_list = json.load(takes_file)
            for take in takes_list:
                if take['take_uid'] in split_set and take['take_uid'] not in skip_takes:
                    self.entries.append(take)
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        # return (5 2second ego clips, 5 2second exo clips, take_uid)
        keys = self.entries[idx]['frame_aligned_videos'].keys()
        #print(keys)
        #print(idx)
        #print(self.entries[idx]['frame_aligned_videos'] )
        for key in keys:
            if 'aria' in key or 'Aria' in key:
                aria_keyname = key
                break
        ego_video_filename = self.data_root_dir + "/" + self.entries[idx]['root_dir'] + '/' \
        +  self.entries[idx]['frame_aligned_videos'][aria_keyname]['rgb']['relative_path']
        if self.entries[idx]['best_exo']:
            exo_keyname = self.entries[idx]['best_exo']
        else:
            if 'cam01' in self.entries[idx]['frame_aligned_videos']:
                exo_keyname = 'cam01'
            elif 'gp01' in self.entries[idx]['frame_aligned_videos']:
                exo_keyname = 'gp01'
            else:
                keys = [key for key in self.entries[idx]['frame_aligned_videos'].keys()]
                keys.remove('collage')
                keys.remove(aria_keyname)
                keys.remove('best_exo')
                exo_keyname = keys[0]
        exo_video_filename = self.data_root_dir + "/" + self.entries[idx]['root_dir'] + '/' \
        +  self.entries[idx]['frame_aligned_videos'][exo_keyname]['0']['relative_path']
        # return (ego_video_filename, exo_video_filename, self.entries[idx]['take_uid'])
        ego_video_data = utils.load_and_transform_video_data([ego_video_filename], self.device, 2, 5)[0]
        exo_video_data =  utils.load_and_transform_video_data([exo_video_filename], self.device, 2, 5)[0]
        #print(idx)
        return (ego_video_data, exo_video_data, self.entries[idx]['take_uid'])


class AudioDataset(Dataset):
    def __init__(self, takes_filename, splits_filename, data_root_dir, split='test', device='cuda', skip_takes=[]):
        self.device = device
        self.split = split
        self.data_root_dir = data_root_dir
        with open(splits_filename) as splits_file:
            splits_map = json.load(splits_file)
        split_set = set(splits_map["split_to_take_uids"][split])
        del splits_map
        self.entries = {}
        with open(takes_filename) as takes_file:
            takes_list = json.load(takes_file)
            for take in takes_list:
                if take['take_uid'] in split_set and take['take_uid'] not in skip_takes:
                    self.entries[take['take_uid']] = take

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, take_uid):
        # return (5 2second ego clips, 5 2second exo clips, take_uid)
        entry = self.entries[take_uid]
        expected_audio_path = f"./audio/{self.split}/{take_uid}.wav"
        if not Path(expected_audio_path).exists():
            collage_path = self.data_root_dir + '/' + entry['root_dir'] + '/' + entry['frame_aligned_videos']['collage']['0']['relative_path']
            print(expected_audio_path)
            print(collage_path)
            subprocess.run(['ffmpeg', '-i', collage_path, '-vn', '-ac', '1', expected_audio_path])
        return utils.load_and_transform_audio_data([expected_audio_path], self.device, clips_per_video=5)
