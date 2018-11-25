import torchvision.transforms as transforms

import config as cf
from autoaugment import DogBreedPolicy

def transform_training():

    transform_train = transforms.Compose([
        transforms.Resize(227),
        #transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        #DogBreedPolicy(),
        transforms.ToTensor(),
    ])  # meanstd transformation

    return transform_train

def transform_testing():

    transform_test = transforms.Compose([
        transforms.Resize(227,227),
        #transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        # CIFAR10Policy(),
        transforms.ToTensor(),
    ])

    return transform_test
