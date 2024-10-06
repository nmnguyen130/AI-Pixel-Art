import torch
import torch.nn as nn
import torch.optim as optim
from .generator import Generator
from .discriminator import Discriminator

class GAN:
    def __init__(self, lr=0.0002, beta1=0.5):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.criterion = nn.BCELoss()
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    def train(self, dataloader, epochs=1000, device='cpu'):
        self.generator.to(device)
        self.discriminator.to(device)
        self.criterion.to(device)

        for epoch in range(epochs):
            for i, (imgs, _) in enumerate(dataloader):
                imgs = imgs.to(device)
                
                # Create labels
                batch_size = imgs.size(0)
                real_labels = torch.ones(batch_size, 1, 1, 1).to(device)
                fake_labels = torch.zeros(batch_size, 1, 1, 1).to(device)
                
                # Train Discriminator
                self.discriminator.zero_grad()

                # Train on real images
                outputs = self.discriminator(imgs, imgs)
                d_loss_real = self.criterion(outputs, real_labels)
                d_loss_real.backward()

                # Train on fake images
                z = torch.randn(batch_size, 4, 64, 64).to(device)  # Assuming z is of shape [batch_size, 4, 64, 64]
                fake_imgs = self.generator(z)
                outputs = self.discriminator(fake_imgs.detach(), fake_imgs.detach())
                d_loss_fake = self.criterion(outputs, fake_labels)
                d_loss_fake.backward()

                d_loss = d_loss_real + d_loss_fake
                self.optimizer_d.step()

                # Train Generator
                self.generator.zero_grad()

                outputs = self.discriminator(fake_imgs, fake_imgs)
                g_loss = self.criterion(outputs, real_labels)
                g_loss.backward()
                self.optimizer_g.step()

                if i % 100 == 0:  # Print the progress every 100 batches
                    print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], '
                          f'D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')
            
        torch.save(self.generator.state_dict(), 'generator.pth')
        torch.save(self.discriminator.state_dict(), 'discriminator.pth')
