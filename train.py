import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import config
from generator_model import Genenator
from discriminator_model import Discriminator
from tqdm import tqdm
import utils

def test_fn(gen, disc, loader, criterion, id):
    gen.eval()
    disc.eval()

    tot_G = 0
    tot_D = 0
    num = 0

    for idx, (data, label) in enumerate(loader):

        data = data.to(config.DEVICE)
        num += data.shape[0]
        cropped_data = data.clone()
        cropped_data[ : , : , 32 : 96, 32 : 96] = 0

        # train disc
        real_center = data[ : , : , 30 : 98, 30 : 98].clone()
        output_real = disc(real_center).squeeze(-1)

        fake_center = gen(cropped_data)
        output_fake = disc(fake_center).squeeze(-1)

        D_real_loss = criterion(output_real, torch.ones_like(output_real))
        D_fake_loss = criterion(output_fake, torch.zeros_like(output_fake))

        tot_D += (D_real_loss + D_fake_loss) / 2

        # train gen
        fake_center = gen(cropped_data)
        output_fake = disc(fake_center).squeeze(-1)
        G_ADV_loss = criterion(output_fake, torch.ones_like(output_fake))

        d_center = (fake_center - real_center).pow(2) * 100
        d_center[:, :, 2 : 66, 2 : 66] -= (fake_center - real_center)[:, :, 2 : 66, 2 : 66].pow(2) * 99

        G_REC_loss = d_center.mean()

        tot_G += config.LAMBDA_ADV * G_ADV_loss + config.LAMBDA_REC * G_REC_loss

    print(f"TEST{id} : d_loss = {tot_D / num : .7f}, g_loss = {tot_G / num: .7f}")
    
    gen.train()
    disc.train()


def train_fn(gen, disc, opt_gen, opt_disc, loader, criterion, id):

    tot_G = 0
    tot_D = 0
    num = 0

    for idx, (data, label) in enumerate(loader):

        data = data.to(config.DEVICE)
        num += data.shape[0]
        cropped_data = data.clone()
        cropped_data[ : , : , 32 : 96, 32 : 96] = 0

        # train disc
        real_center = data[ : , : , 30 : 98, 30 : 98].clone()
        output_real = disc(real_center).squeeze(-1)

        fake_center = gen(cropped_data)
        output_fake = disc(fake_center).squeeze(-1)

        D_real_loss = criterion(output_real, torch.ones_like(output_real))
        D_fake_loss = criterion(output_fake, torch.zeros_like(output_fake))

        D_loss = (D_real_loss + D_fake_loss) / 2
        tot_D += D_loss

        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

        # train gen
        fake_center = gen(cropped_data)
        output_fake = disc(fake_center).squeeze(-1)
        G_ADV_loss = criterion(output_fake, torch.ones_like(output_fake))

        d_center = (fake_center - real_center).pow(2) * 100
        d_center[:, :, 2 : 66, 2 : 66] -= (fake_center - real_center)[:, :, 2 : 66, 2 : 66].pow(2) * 99

        G_REC_loss = d_center.mean()

        G_loss = config.LAMBDA_ADV * G_ADV_loss + config.LAMBDA_REC * G_REC_loss
        tot_G += G_loss

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()
        
    cropped_data[ : , : , 32 : 96, 32 : 96] = fake_center[:, :, 2 : 66, 2 : 66]
    utils.save_img("1.png", data[0])
    utils.save_img("2.png", cropped_data[0])
    print(f"Echo{id}")
    print(f"TRAIN{id} : d_loss = {D_loss / num : .7f}, g_loss = {G_loss / num : .7f}")
    print()
    
    if config.SAVE_MODEL:
        utils.save_checkpoint(gen, opt_gen, config.CHECKPOINT_G)
        utils.save_checkpoint(disc, opt_disc, config.CHECKPOINT_D)


def main():
    # dataloader
    train_dataset = datasets.ImageFolder(
        root = config.TRAIN_DIR,
        transform = config.transform
    )

    test_dataset = datasets.ImageFolder(
        root = config.TEST_DIR,
        transform = config.transform
    )

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = config.BATCH_SIZE,
        shuffle = True,
        num_workers = config.NUM_WORKER
    )

    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = config.BATCH_SIZE,
        shuffle = True,
        num_workers = config.NUM_WORKER
    )

    # model
    disc = Discriminator().to(config.DEVICE)
    gen = Genenator().to(config.DEVICE)

    # optimizer
    opt_disc = optim.Adam(params = disc.parameters(), lr = config.LEARNING_RATE_D)
    opt_gen = optim.Adam(params = gen.parameters(), lr = config.LEARNING_RATE_G)

    # criterion
    criterion = nn.BCELoss()

    if config.LOAD_MODEL:
        utils.load_checkpoint(gen, opt_gen, config.CHECKPOINT_G)
        utils.load_checkpoint(disc, opt_disc, config.CHECKPOINT_D)

    for i  in range(config.NUM_EPOCHS):
        train_fn(gen, disc, opt_gen, opt_disc, train_loader, criterion, i)
        test_fn(gen, disc, test_loader, criterion, i)


if __name__ == '__main__':
    main()