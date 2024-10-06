import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Encoder
        self.enc1 = self.encoder_block(4, 64, normalization=False)
        self.enc2 = self.encoder_block(64, 128)
        self.enc3 = self.encoder_block(128, 256)
        self.enc4 = self.encoder_block(256, 512)
        self.enc5 = self.encoder_block(512, 512)
        self.enc6 = self.encoder_block(512, 512, normalization=False)

        # Decoder
        self.dec1 = self.decoder_block(512, 1024, dropout=True)
        self.dec2 = self.decoder_block(1024 + 512, 1024, dropout=True)
        self.dec3 = self.decoder_block(1024 + 512, 512, dropout=True)
        self.dec4 = self.decoder_block(512 + 256, 256)
        self.dec5 = self.decoder_block(256 + 128, 128)
        self.dec6 = self.decoder_block(128 + 64, 64)

        # Final output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 4, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def encoder_block(self, in_channels, out_channels, normalization=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if normalization:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def decoder_block(self, in_channels, out_channels, dropout=False):
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                  nn.InstanceNorm2d(out_channels),
                  nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)

        # Decoder + Skip Connections
        dec1 = self.dec1(enc6)
        dec2 = self.dec2(torch.cat((dec1, enc5), dim=1))
        dec3 = self.dec3(torch.cat((dec2, enc4), dim=1))
        dec4 = self.dec4(torch.cat((dec3, enc3), dim=1))
        dec5 = self.dec5(torch.cat((dec4, enc2), dim=1))
        dec6 = self.dec6(torch.cat((dec5, enc1), dim=1))
        
        return self.final(dec6)
    
if __name__ == "__main__":
    model = Generator()
    input_tensor = torch.randn(1, 4, 64, 64) # Kích thước đầu vào: [batch_size, channels, height, width]
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)