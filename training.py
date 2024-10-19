import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from keras import metrics, callbacks
from datetime import datetime
from model import AtrousGSqueezeSeg
import pandas as pd


@keras.utils.register_keras_serializable(package="my_package", name="dice_coef")
def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)


@keras.utils.register_keras_serializable(package="my_package", name="dice_coef_loss")
def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def mean_iou(y_true, y_pred):
    yt0 = y_true[:, :, :, 0]
    yp0 = tf.keras.backend.cast(y_pred[:, :, :, 0] > 0.5, 'float32')
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter / union, 'float32'))
    return iou


#ultrascale+ ZCU102
def train(size=(128, 128), num_class=2, basepath="dataset/"):

    x_train_name = f"{basepath}X_train_{size[1]}"
    y_train_name = f"{basepath}Y_train_{size[1]}"
    x_val_name = f"{basepath}X_valid_{size[1]}"
    y_val_name = f"{basepath}Y_valid_{size[1]}"

    X_train = np.load(x_train_name + ".npy")
    Y_train = np.load(y_train_name + ".npy")
    X_val = np.load(x_val_name + ".npy")
    Y_val = np.load(y_val_name + ".npy")

    Y_train = tf.one_hot(tf.cast(tf.squeeze(Y_train, axis=-1), dtype=tf.int32), depth=num_class)
    Y_val = tf.one_hot(tf.cast(tf.squeeze(Y_val, axis=-1), dtype=tf.int32), depth=num_class)


    num_color_channels = 1
    #model = ld_unet(size + (num_color_channels,), num_class)
    model = AtrousGSqueezeSeg(size + (num_color_channels,), num_class)
    #model = ld_unet(size + (1,))
    model.summary()
    #model = get_model(size, num_class, [16, 32], [32, 16, 2])
    filename = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    csv_logger = tf.keras.callbacks.CSVLogger(f'results/metrics_{filename}.csv')

    model_path = f'results/best_model_{filename}'

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = model_path + '.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    # Compilare il modello con diverse metriche
    model.compile(optimizer=keras.optimizers.Adam(1e-4),  #Adam(0.002),#"rmsprop",
                  loss=dice_coef_loss,
                  metrics=[
                      metrics.CategoricalAccuracy(name='accuracy'),
                      dice_coef,
                      #metrics.AUC(name='auc'),
                      metrics.Precision(name='precision'),
                      metrics.Precision(name='precision_c0', class_id=0),
                      metrics.Precision(name='precision_c1', class_id=1),
                      metrics.Recall(name='recall'),
                      metrics.Recall(name='recall_c0', class_id=0),
                      metrics.Recall(name='recall_c1', class_id=1),
                      metrics.MeanIoU(num_classes=num_class, name='mean_iou'),
                      metrics.OneHotIoU(num_classes=num_class, target_class_ids=[0, 1],
                                        name='one_hot_iou'),
                      metrics.OneHotMeanIoU(num_classes=num_class, name='one_hot_mean_iou'),
                      #metrics.F1Score(average="weighted", name='f1_score')  # F1 score
                  ])
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.35, patience=5, min_lr=1e-6)

    # Addestrare il modello
    history = model.fit(np.concatenate((X_train, X_val), axis=0),
                        np.concatenate((Y_train, Y_val), axis=0),
                        epochs=50,
                        batch_size=32,
                        callbacks=[csv_logger, checkpoint_callback, reduce_lr],
                        #validation_data=(X_val, Y_val)
                        validation_split=0.2,
                        )
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 20))

    axes[0].plot(history.history['accuracy'])
    axes[0].plot(history.history['val_accuracy'])
    axes[0].set_title('model accuracy')
    axes[0].set_label('accuracy')
    axes[0].set_xlabel('epoch')
    axes[0].legend(['train', 'val'], loc='upper left')
    axes[1].plot(history.history['loss'])
    axes[1].plot(history.history['val_loss'])
    axes[1].set_title('model loss')
    axes[1].set_label('loss')
    axes[1].set_xlabel('epoch')
    axes[1].legend(['train', 'val'], loc='upper left')
    axes[2].plot(history.history['dice_coef'])
    axes[2].plot(history.history['val_dice_coef'])
    axes[2].set_title('model dice coef')
    axes[2].set_label('dice_coef')
    axes[2].set_xlabel('epoch')
    axes[2].legend(['train', 'val'], loc='upper left')
    plt.show()
    return model, f'model_{filename}'