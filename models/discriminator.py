import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(8, 64, 4, stride=2, padding=1),  # Input: 8 channels (4 input + 4 condition)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=0),  # Change stride to 1 to avoid excessive downsampling
            nn.Sigmoid()
        )

    def forward(self, x, condition):
        x = torch.cat((x, condition), dim=1)  # Nối input với điều kiện
        return self.model(x)
    
if __name__ == "__main__":
    model = Discriminator()
    input_tensor = torch.randn(1, 4, 64, 64)  # Kích thước đầu vào: [batch_size, channels, height, width]
    condition_tensor = torch.randn(1, 4, 64, 64)  # Kích thước điều kiện: [batch_size, channels, height, width]
    output_tensor = model(input_tensor, condition_tensor)
    print("Output tensor shape:", output_tensor.shape)