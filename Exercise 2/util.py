from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import torch

class DogDataset(Dataset):
    """Dog Breed Dataset"""

    def __init__(self, filenames,labels,root_dir,transform=None):
        assert len(filenames)==len(labels)
        self.filenames = filenames
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        label = self.labels[item]
        img_name = os.path.join(self.root_dir,self.filenames[item]+'.jpg')

        with Image.open(img_name) as f:
            img = f.convert('RGB')

        if self.transform:
            img = self.transform(img)

        if self.labels is None:
            return img,self.filenames[item]
        else:
            return img,self.labels[item]

def get_train_dataset(filenames,labels,batch_size,rootdir='./data/dog-breed/train'):
    composed = transforms.Compose([transforms.RandomResizedCrop(227, scale=(0.75, 1)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                                   ])
    dog_trainset = DogDataset(filenames, labels, transform=composed,root_dir=rootdir)
    #dog_train = DataLoader(dog_trainset, batch_size, True)
    return dog_trainset


def get_test_dataset(filenames,batch_size,rootdir='./data/dog-breed/test'):
    composed = transforms.Compose([transforms.Resize(227),
                                    transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                                   ])
    dog_testset = DogDataset(filenames,None,transform=composed,root_dir=rootdir)
    #dog_test = DataLoader(dog_testset, batch_size, False)
    return dog_testset
