import torch
from torch.utils.data import Dataset
import os
import numpy as np
import json
from einops import rearrange

    
def get_subfolders(path):
    all_items = os.listdir(path)
    return [name for name in all_items 
            if os.path.isdir(os.path.join(path, name))]

def get_npy_files(directory):
    npy_files = []
    if not os.path.exists(directory):
        print(f"❌: {directory}")
        return []

    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            npy_files.append(filename[:-4])
    return npy_files


def load_yield_data(file_path):
    yield_data = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    fip = data['fip']
                    yield_label = data['yield label']
                    yield_data[fip] = yield_label
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"❌ {line_num} error: {e}")
    
    return yield_data


def get_yield_by_fip(fip, yield_data):

    if fip in yield_data:
        return yield_data[fip]
    else:
        return None


class Sentinel_Dataset(Dataset):

    def __init__(self, image_dir, label_path):

        self.directory = image_dir
        self.fips_codes = get_npy_files(image_dir)
        self.label = load_yield_data(label_path)

    def __len__(self):
        return len(self.fips_codes)

    def __getitem__(self, index):
        fips_code = self.fips_codes[index]

        file_path = self.directory+"/{}.npy".format(fips_code)
        data = np.load(file_path)
        images = torch.tensor(data, dtype=torch.float32)
        images = rearrange(images, 'g t h w c -> t g h w c')  

        yield_label = get_yield_by_fip(fips_code, self.label)
        yield_label = torch.tensor(yield_label[0])

        return images,yield_label


        
if __name__ == '__main__':
    image_dir = "yieldpredicion/data"
    label_path = "yieldpredicion/yieldlabel.txt"
    dataset = Sentinel_Dataset(image_dir, label_path)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    i=0
    for images,yield_label in train_loader:
        i=i+1
    print(i)
