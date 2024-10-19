import numpy as np


def get_stat(basepath="dataset/", size=(128, 128)):
    train = np.load(f"{basepath}X_train_{size[1]}.npy")
    valid = np.load(f"{basepath}X_valid_{size[1]}.npy")
    test = np.load(f"{basepath}X_test_{size[1]}.npy")
    dataset = np.concatenate((train, valid, test), axis=0)
    mean = np.mean(dataset)
    var = np.var(dataset)
    print(f"dataset mean: {mean}")
    print(f"dataset var: {var}")
