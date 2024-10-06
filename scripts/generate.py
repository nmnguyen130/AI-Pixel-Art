import torch
from torchvision import transforms
from models.generator import Generator
from PIL import Image

def generate_sprite(generator, input_image_path, output_image_path):
    input_image = Image.open(input_image_path).convert('RGBA')
    input_tensor = transforms.ToTensor()(input_image).unsqueeze(0)
    generator.eval()
    with torch.no_grad():
        output_tensor = generator(input_tensor)
    output_image = transforms.ToPILImage()(output_tensor.squeeze(0))
    output_image.save(output_image_path)

if __name__ == "__main__":
    generator = Generator()
    generator.load_state_dict(torch.load('generator.pth'))
    generate_sprite(generator, 'input.png', 'output.png')
