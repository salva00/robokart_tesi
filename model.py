import tensorflow as tf
from tensorflow import keras
from keras import layers, models
#@keras.utils.register_keras_serializable(package="my_package", name="AtrousGFire")
def AtrousGFire(squeeze_channels, expand1x1_channels, expand3x3_channels, dilation_rate):
    def layer(x):
        squeeze = layers.SeparableConv2D(squeeze_channels, (1, 1))(x)
        squeeze = layers.BatchNormalization()(squeeze)
        squeeze = layers.Activation('relu')(squeeze)

        expand1x1 = layers.SeparableConv2D(expand1x1_channels, (1, 1))(squeeze)
        expand1x1 = layers.BatchNormalization()(expand1x1)
        expand1x1 = layers.Activation('relu')(expand1x1)

        expand3x3 = layers.SeparableConv2D(expand3x3_channels, (3, 3), padding='same', dilation_rate=dilation_rate)(squeeze)
        expand3x3 = layers.BatchNormalization()(expand3x3)
        expand3x3 = layers.Activation('relu')(expand3x3)

        return layers.Concatenate()([expand1x1, expand3x3])

    return layer

#@keras.utils.register_keras_serializable(package="my_package", name="AtrousGSqueezeSeg")
def AtrousGSqueezeSeg(input_shape, num_classes=3):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.SeparableConv2D(input_shape[0]//2, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = AtrousGFire(input_shape[0]//8, input_shape[0]//4, input_shape[0]//4, dilation_rate=2)(x)
    x = AtrousGFire(input_shape[0]//8, input_shape[0]//4, input_shape[0]//4, dilation_rate=3)(x)
    x = AtrousGFire(input_shape[0]//8, input_shape[0]//4, input_shape[0]//4, dilation_rate=5)(x)
    #x = AtrousGFire(input_shape[0]//8, input_shape[0]//4, input_shape[0]//4, dilation_rate=8)(x)

    x = layers.SeparableConv2D(num_classes, (1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    outputs = layers.Activation('softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model