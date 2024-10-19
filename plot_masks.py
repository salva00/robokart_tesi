import keras
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns

from training import dice_coef_loss

labs = ['bckgrnd', 'road']
cols = ['b', 'g']

cmap = ListedColormap(cols)


def plt_prediction(X_true, Y_true, Y_pred, num_class=2):
    y_pred_p = np.array(tf.math.argmax(Y_pred, axis=-1))
    Y_true_onehot = tf.one_hot(tf.cast(tf.squeeze(Y_true, axis=-1), dtype=tf.int32), depth=num_class)
    Y_true_labels = np.argmax(Y_true_onehot, axis=-1).flatten()
    Y_pred_labels = np.argmax(Y_pred, axis=-1).flatten()
    print(f"Y_true_onehot: {Y_true_onehot.shape}, Y_pred: {Y_pred.shape}")
    # Calcolare la matrice di confusione
    cm = confusion_matrix(Y_true_labels, Y_pred_labels)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 20))
    axes[0].imshow(X_true[:, :, 0], cmap=plt.cm.gray)  #plot just the first channel in greyscale
    axes[0].set_title("Ground Truth")
    im1 = axes[0].imshow(Y_true, cmap=cmap, alpha=0.3, vmin=0,
                         vmax=num_class - 1)  #plot mask with 50% transparency
    cbar = plt.colorbar(im1, shrink=0.5, ax=axes[0],
                        ticks=np.arange(num_class))  # make a colorbar, make it smaller and make integer tick spacing
    cbar.ax.set_yticklabels(labs)  #tick labels for the colorbar

    axes[1].set_title("Predicted mask")
    axes[1].imshow(X_true[:, :, 0], cmap=plt.cm.gray)  # plot just the first channel in greyscale
    im2 = axes[1].imshow(y_pred_p[-1, :, :], cmap=cmap, alpha=0.3, vmin=0,
                         vmax=num_class - 1)  # plot mask with 50% transparency
    cbar = plt.colorbar(im2, shrink=0.5, ax=axes[1],
                        ticks=np.arange(num_class))  # make a colorbar, make it smaller and make integer tick spacing
    cbar.ax.set_yticklabels(labs)  # tick labels for the colorbar

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2])
    axes[2].set_xlabel('Predicted Labels')
    axes[2].set_ylabel('True Labels')
    axes[2].set_title('Confusion Matrix')

    plt.show()
    plt.close()