import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
from PIL import Image
from torch import optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from glob import glob
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
import numpy as np
from torchvision.transforms import transforms
from tqdm import tqdm
import os

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

csv_path = '<define path-to-csv file here>'
img_dir = '<define path-to-image-folder here'

class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.pairs = list()

        for idx in tqdm(range(len(annotations_file))):
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            image = Image.open(img_path).convert('RGB') ## this is being done because before we were dealing just with greyscale images 😦
            label = self.img_labels.iloc[idx, 1]

            image = self.transform(image)
            self.pairs.append((image, label))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        image, label = pair[0], pair[1]
        return image, label



df = pd.read_csv(csv_path)
# this command returns the same old column (species) but instead of having the name of the species, 
## we have the number associated to each species
new_species = df['species'].astype('category').cat.codes  


# Then, we overwrite the old column, with this brand new numerical column
df['species'] = new_species


DATASET = ImageDataset(df, img_dir, img_transform)

val_size = int(len(DATASET) - len(DATASET)*0.8)

image_train, image_val = torch.utils.data.random_split(DATA, (len(DATA)- val_size, val_size)) 


''' Dealing with Class imbalance '''


Y = list()
for i in range(len(image_train)):
    Y.append(image_train[i][1])
    
labels_on_train = pd.Series(Y)
unique_new_species = labels_on_train.unique()
train_labels_np = np.array(labels_on_train)


import torch.utils.data as data
from collections import Counter



class_counts = Counter(labels_on_train)
n_samples = len(train_labels_np)

weights = [1.0 / class_counts[x] for x in train_labels_np]


sampler = data.WeightedRandomSampler(weights, n_samples)

train_loader = DataLoader(image_train, batch_size=32, sampler=sampler)
val_loader = DataLoader(image_val, batch_size=32)




