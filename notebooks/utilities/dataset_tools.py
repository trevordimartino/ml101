import torch
import torchvision


def load_dice_dataset(sub_folder='train', batch_size=16):
    data_path = f'../data/dice/{sub_folder}/'
    dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )
    return dataset, loader


def load_test_dice_set(sub_folder='wut', batch_size=16):
    data_path = f'../data/dice/{sub_folder}/'
    transforms_chain = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((480, 480)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transforms_chain
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )
    return dataset, loader
