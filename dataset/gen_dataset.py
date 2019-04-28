import torch.utils.data as data
import torch
from PIL import Image, ImageFile
import os
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IAMGES = True


# https://github.com/pytorch/vision/issues/81

transform_totensor = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])

def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(path).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)


def default_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath = line.strip()
            imgList.append((imgPath))
    return imgList


class ImageList(data.Dataset):
    def __init__(self, root, fileList, transform=None, list_reader=default_reader, loader=PIL_loader):
        self.root = root
        self.imgList = list_reader(fileList)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        imgPath = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))
        # img = img.resize((320,240))
        # print(img.getpixel((0,0)))
        if self.transform is not None:
            img = self.transform(img)
        # img = transforms.ToTensor(img)
        # img = torch.Tensor(img)
        return img, imgPath

    def __len__(self):
        return len(self.imgList)
