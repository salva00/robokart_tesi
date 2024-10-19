from PIL import Image
import cv2
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

LABEL_NAMES = ['background', 'road']

COLOR_MAPPING = {
    0: (0, 0, 0),  # Background
    1: (184, 61, 245),  # Person
}


def convert_mask(mask_rgb):
    mask = np.array(mask_rgb)
    new_mask = np.zeros(shape=(mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            a = mask[row, col, :]
            final_key = None
            final_d = None
            for key, value in COLOR_MAPPING.items():
                d = np.sum(np.sqrt(pow(a - value, 2)))
                if final_key is None:
                    final_d = d
                    final_key = key
                elif d < final_d:
                    final_d = d
                    final_key = key
            new_mask[row, col] = final_key
    new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1], 1))
    return new_mask


def show_example(size=(128, 128), num_class=2, basepath="dataset/", fov=110):
    X_train = np.load(f"{basepath}X_train_{size[1]}.npy")
    Y_train = np.load(f"{basepath}Y_train_{size[1]}.npy")
    for i in range(3):
        img = X_train[i]
        mask = Y_train[i]
        # Create a figure with 2 subplots (1 row, 2 columns)
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))

        # Display the images in the subplots
        axs[0].imshow(img)
        axs[0].set_title('Original Image')

        axs[1].imshow(mask, cmap='gray', alpha=0.6)
        axs[1].set_title('Mask Image')
        # Remove the x and y ticks
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.show()
        plt.close()


def preprocess(size=(128, 128), num_class=2, basepath="dataset/", bw=False, version=3):
    x_train_name = f"{basepath}X_train_{size[1]}"
    y_train_name = f"{basepath}Y_train_{size[1]}"
    dataset_dirs = ['test', 'train', 'valid']
    if (os.path.isfile(x_train_name + ".npy") and
            os.path.isfile(y_train_name + ".npy")):
        show_example(size, num_class, basepath=basepath)
        return
    if version >= 3:
        for split in dataset_dirs:
            X = []
            Y = []
            x_name = f"{basepath}X_{split}_{size[1]}"
            y_name = f"{basepath}Y_{split}_{size[1]}"
            image_path = basepath + split + '/imgs/'
            mask_path = basepath + split + '/masks/'
            for file in tqdm(os.listdir(image_path)):
                img = Image.open(image_path + file).resize(size)
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
                img_normalized = img / 255.0
                base_name, ext = os.path.splitext(mask_path + file)
                new_filename = base_name + ".png"
                mask = Image.open(new_filename).resize(size)
                coverted_mask = convert_mask(mask)
                X.append(img_normalized)
                Y.append(coverted_mask)
            np.save(x_name + ".npy", np.array(X))
            np.save(y_name + ".npy", np.array(Y))
    show_example(size, num_class, basepath=basepath)
