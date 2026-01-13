import h5py
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_data():
    train_dataset = h5py.File('datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) # your train set features
    y_train = np.array(train_dataset["Y_train"][:]) # your train set labels

    test_dataset = h5py.File('datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) # your train set features
    y_test = np.array(test_dataset["Y_test"][:]) # your train set labels
    
    return X_train, y_train, X_test, y_test




def load_data_large():
    Images_cat = 'PetImages/Cat'
    Images_dog = 'PetImages/Dog'

    X_train_cat, X_train_dog = [], []
    y_train_cat, y_train_dog = [], []

    img_size= (64, 64)

    for i, filename in tqdm(enumerate(os.listdir(Images_cat))):
        fpath = os.path.join(Images_cat, filename)
        img = Image.open(fpath)
        img = img.resize(img_size)
        img = img.convert("RGB").resize(img_size, Image.LANCZOS)
        X_train_cat.append(np.array(img))
        y_train_cat.append(0)

    for i, filename in tqdm(enumerate(os.listdir(Images_dog))):
        fpath = os.path.join(Images_dog, filename)
        img = Image.open(fpath)
        img = img.resize(img_size)
        img = img.convert("RGB").resize(img_size, Image.LANCZOS)
        X_train_dog.append(np.array(img))
        y_train_dog.append(1)

    X = np.array(X_train_cat + X_train_dog)
    y = np.array(y_train_cat + y_train_dog)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    return X_train, X_test, y_train, y_test





