import torch
import torchvision
import sys
from transform import transform_training, transform_testing
import config as cf
#changes start here

from util import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.label import LabelEncoder

#changes end here

def dataset(dataset_name):

    if (dataset_name == 'cifar10'):
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_training())
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_testing())
        outputs = 10
        inputs = 3
    
    elif (dataset_name == 'cifar100'):
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_training())
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_testing())
        outputs = 100
        inputs = 3
    
    elif (dataset_name == 'mnist'):
        print("| Preparing MNIST dataset...")
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_training())
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_testing())
        outputs = 10
        inputs = 1
    
    elif (dataset_name == 'fashionmnist'):
        print("| Preparing FASHIONMNIST dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_training())
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_testing())
        outputs = 10
        inputs = 1
    elif (dataset_name == 'stl10'):
        print("| Preparing STL10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.STL10(root='./data',  split='train', download=True, transform=transform_training())
        testset = torchvision.datasets.STL10(root='./data',  split='test', download=False, transform=transform_testing())
        outputs = 10
        inputs = 3
        
    elif (dataset_name == 'dog-breed'):
        print("| Preparing DOG-BREED dataset...")
        
        data_train_csv = pd.read_csv('./data/dog-breed/labels.csv')
        filenames = data_train_csv.id.values
        le = LabelEncoder()
        labels = le.fit_transform(data_train_csv.breed)

        filenames_train , filenames_val ,labels_train, labels_val =train_test_split(filenames,labels,test_size=0.3,stratify=labels,shuffle=True)
        trainset = get_train_dataset(filenames_train,labels_train,cf.batch_size,rootdir='./data/dog-breed/train')
        testset = get_train_dataset(filenames_val,labels_val,cf.batch_size,rootdir='./data/dog-breed/train')
        outputs = 120
        inputs = 3
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cf.batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cf.batch_size, shuffle=True, num_workers=4)
    
    return trainloader, testloader, outputs, inputs

