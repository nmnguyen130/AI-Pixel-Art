import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.gan import GAN

class RGBAImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # Duyệt qua các thư mục con và lấy tất cả các tệp ảnh PNG
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.png'):
                    self.image_paths.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGBA')
        if self.transform:
            image = self.transform(image)
        return image, 0  # Trả về một giá trị nhãn giả

def main():
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
    ])

    dataset = RGBAImageDataset('data/raw', transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    gan = GAN()
    gan.train(dataloader)

if __name__ == "__main__":
    main()
