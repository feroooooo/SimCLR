import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR

if __name__ == "__main__":
    dataset = ContrastiveLearningDataset("./datasets")
    train_dataset = dataset.get_dataset("stl10", 2)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True, drop_last=True)
    model = ResNetSimCLR(base_model="resnet18", out_dim=128)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    gpu_index = 0
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        gpu_index = -1
    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)