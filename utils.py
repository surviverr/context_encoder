from PIL import Image
import torch
import config
import numpy as np

def save_img(filename, data):
    tmp = data.detach().cpu().numpy().transpose(1, 2, 0)
    tmp = (tmp + 1) / 2 * 255
    img = Image.fromarray(tmp.astype(np.uint8))
    img.save(filename)

def load_img(filename):
    return Image.open(filename)

def save_checkpoint(model, optimizer, filename):
    checkpoint = {
        "model" : model.state_dict(),
        "optimizer" : optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename, map_location = config.DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

if __name__ == '__main__':
    x = torch.zeros((3, 2, 3))
    save_img("1.png", x)