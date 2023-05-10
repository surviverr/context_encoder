import utils
import config
import argparse
from generator_model import Genenator
import torch.optim as optim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help = "the path of img")
    opt = parser.parse_args()
    x = utils.load_img(opt.path)
    x = config.transform(x).to(config.DEVICE)
    gen = Genenator().to(config.DEVICE)
    opt_gen = optim.Adam(params = gen.parameters(), lr =  config.LEARNING_RATE_G)
    utils.load_checkpoint(gen, opt_gen, config.CHECKPOINT_G)
    x = x.reshape(1, x.shape[0], x.shape[1], -1)
    gen.eval()
    y = gen(x)
    utils.save_img("result/real.png", x[0])
    x[:, :, 32 : 96, 32 : 96] = 0
    utils.save_img("result/cropped.png", x[0])
    x[:, :, 32 : 96, 32 : 96] = y[:, :, 2 : 66, 2 : 66]
    utils.save_img("result/fake.png", x[0])
    

if __name__ == '__main__':
    main()