from torch.utils.data import DataLoader
from os.path import join
from torchvision.transforms import Compose, Resize, Normalize, ToTensor


def load_train_data(batch_size=128, sampler=None):
    cuda = True
    loader_kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
    
    train_loader = pcap 파일 읽기

    return train_loader



