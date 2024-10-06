import torch
import os
from numpy import np
from PIL import Image
from torch.utils.data import Dataset

class PixelArtDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = os.listdir(data_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGBA")
        image = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
        return image

def load_data(data_dir):
    return PixelArtDataset(data_dir)

def save_model(model, path):
    torch.save(model.state_dict(), path)
