import tensorflow as tf
num_epochs=2
num_class = 2  # Cambia questo valore a seconda del tuo dataset
img_size = (64, 64, 1)
dataset = f"110"
X_train_path = f"./dataset/seg/{dataset}/v3/X_train_{img_size[0]}.npy"
Y_train_path = f"./dataset/seg/{dataset}/v3/Y_train_{img_size[0]}.npy"
X_val_path = f"./dataset/seg/{dataset}/v3/X_valid_{img_size[0]}.npy"
Y_val_path = f"./dataset/seg/{dataset}/v3/Y_valid_{img_size[0]}.npy"
X_test_path = f"./dataset/seg/{dataset}/v3/X_test_{img_size[0]}.npy"
Y_test_path = f"./dataset/seg/{dataset}/v3/Y_test_{img_size[0]}.npy"
num_batch = 16

from tensorflow import keras
from keras import layers, models, utils, callbacks, optimizers
from keras import metrics
import numpy as np


@keras.utils.register_keras_serializable(package="my_package", name="dice_coef")
def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)


@keras.utils.register_keras_serializable(package="my_package", name="dice_coef_loss")
def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

class BestModelSaver(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_accuracy', mode='max'):
        super(BestModelSaver, self).__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_weights = None
        self.best = -np.Inf if mode == 'max' else np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is not None:
            if (self.mode == 'max' and current > self.best) or (self.mode == 'min' and current < self.best):
                self.best = current
                self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)

# Usare il callback
best_model_saver = BestModelSaver(monitor='val_accuracy', mode='max')

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'best_model_cinese.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

def dilation_residual_block(x, filters, dilation_rate):
    dsc_3x3 = layers.SeparableConv2D(filters, kernel_size=3, dilation_rate=dilation_rate, padding='same')(x)
    dsc_1x1 = layers.SeparableConv2D(filters, kernel_size=1, padding='same')(x)

    x = layers.Add()([dsc_3x3, dsc_1x1])
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def ld_unet(input_size=(128, 128, 1), num_classes=3):
    inputs = layers.Input(input_size)


    # Encoder
    c0 = dilation_residual_block(inputs, filters=8, dilation_rate=2)
    p0 = layers.SeparableConv2D(16, (3, 3), strides=(2, 2), padding='same')(c0)

    c1 = dilation_residual_block(p0, filters=16, dilation_rate=2)
    p1 = layers.SeparableConv2D(32, (3, 3), strides=(2, 2), padding='same')(c1)

    c2 = dilation_residual_block(p1, filters=32, dilation_rate=2)
    p2 = layers.SeparableConv2D(64, (3, 3), strides=(2, 2), padding='same')(c2)

    c3 = dilation_residual_block(p2, filters=64, dilation_rate=2)
    p3 = layers.SeparableConv2D(128, (3, 3), strides=(2, 2), padding='same')(c3)

    # Bottleneck
    c4 = dilation_residual_block(p3, filters=128, dilation_rate=2)

    # Decoder
    u1 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(c4)
    u1 = layers.SeparableConv2D(64, (3, 3), padding='same')(u1)
    u1 = layers.Add()([u1, c3])
    u1 = dilation_residual_block(u1, filters=64, dilation_rate=2)

    u2 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(u1)
    u2 = layers.SeparableConv2D(32, (3, 3), padding='same')(u2)
    u2 = layers.Add()([u2, c2])
    u2 = dilation_residual_block(u2, filters=32, dilation_rate=2)

    u3 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(u2)
    u3 = layers.SeparableConv2D(16, (3, 3), padding='same')(u3)
    u3 = layers.Add()([u3, c1])
    u3 = dilation_residual_block(u3, filters=16, dilation_rate=2)

    u4 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(u3)
    u4 = layers.SeparableConv2D(8, (3, 3), padding='same')(u4)
    u4 = layers.Add()([u4, c0])
    u4 = dilation_residual_block(u4, filters=8, dilation_rate=2)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(u4)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

model = ld_unet(img_size,num_classes=num_class)#SameNumOfNeurons#, MoreNeurons]
i = 1

model.compile(optimizer="adam",#keras.optimizers.Adam(1e-4),
              loss=dice_coef_loss,
              metrics=['accuracy',
                       dice_coef,
                        metrics.Precision(name='precision'),
                      metrics.Precision(name='precision_c0', class_id=0),
                      metrics.Precision(name='precision_c1', class_id=1),
                      metrics.Recall(name='recall'),
                      metrics.Recall(name='recall_c0', class_id=0),
                      metrics.Recall(name='recall_c1', class_id=1),
                      metrics.MeanIoU(num_classes=num_class, name='mean_iou'),
                      metrics.OneHotIoU(num_classes=num_class, target_class_ids=[0, 1],
                                        name='one_hot_iou'),
                      metrics.OneHotMeanIoU(num_classes=num_class, name='one_hot_mean_iou')])
csv_logger = keras.callbacks.CSVLogger(f'metrics_cinese_{i}.csv')

# Mostrare il sommario del modello
model.summary()
# Creazione di un tensor di esempio
#example_input = tf.random.normal([1, 224, 224, 3])
#output = model(example_input)
#print(output.shape)  # Dovrebbe essere [1, 14, 14, num_classes] a causa del downsampling

X_train = np.load(X_train_path)
Y_train = np.load(Y_train_path)
X_val = np.load(X_val_path)
Y_val = np.load(Y_val_path)
Y_train = tf.one_hot(tf.cast(tf.squeeze(Y_train, axis=-1), dtype=tf.int32), depth=num_class)
Y_val = tf.one_hot(tf.cast(tf.squeeze(Y_val, axis=-1), dtype=tf.int32), depth=num_class)
checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=f'results/best_model_cinese.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.35, patience=5, min_lr=1e-6)

#history = model.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((Y_train, Y_val), axis=0), validation_split=0.3, shuffle=True, epochs=num_epochs, batch_size=num_batch,verbose=1,callbacks=[best_model_saver,csv_logger,checkpoint_callback,reduce_lr])
history = model.fit(X_train,Y_train, validation_data=(X_val,Y_val), shuffle=True, epochs=num_epochs, batch_size=num_batch,verbose=1,callbacks=[best_model_saver,csv_logger,checkpoint_callback,reduce_lr])
model.set_weights(best_model_saver.best_weights)
i = i + 1

import tensorflow_model_optimization as tfmot

quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer="adam",#keras.optimizers.Adam(1e-4),
              loss=dice_coef_loss,
              metrics=['accuracy',
                       dice_coef,
                        metrics.Precision(name='precision'),
                      metrics.Precision(name='precision_c0', class_id=0),
                      metrics.Precision(name='precision_c1', class_id=1),
                      metrics.Recall(name='recall'),
                      metrics.Recall(name='recall_c0', class_id=0),
                      metrics.Recall(name='recall_c1', class_id=1),
                      metrics.MeanIoU(num_classes=num_class, name='mean_iou'),
                      metrics.OneHotIoU(num_classes=num_class, target_class_ids=[0, 1],
                                        name='one_hot_iou'),
                      metrics.OneHotMeanIoU(num_classes=num_class, name='one_hot_mean_iou')])

q_aware_model.summary()