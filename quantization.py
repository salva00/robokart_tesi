import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, utils


def to_tf_lite(model, name, size=(128, 128), num_class=2, basepath="dataset/"):
    # converto da modello keras a tensorflow lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_models_dir = pathlib.Path('deploy/tflite/')
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir / f"{name}.tflite"
    tflite_model_file.write_bytes(tflite_model)

    # quantizzo il modello a float16
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_fp16_model = converter.convert()
    tflite_model_fp16_file = tflite_models_dir / f"{name}_{size[0]}_{num_class}classes_f16.tflite"
    tflite_model_fp16_file.write_bytes(tflite_fp16_model)

    data_name = f'{basepath}X_test_{size[0]}'

    representative_data = np.load(data_name + '.npy').astype(np.float32)

    def representative_dataset_gen():
        num_color_channels = 1
        for data in representative_data:
            yield [data.reshape(1, size[0], size[1],
                                num_color_channels)]  # Modifica la shape in base ai requisiti del tuo modello

    # quantizzo il modello a int8
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative_dataset_gen
    tflite_i8_model = converter.convert()
    tflite_model_i8_file = tflite_models_dir / f"{name}_{size[0]}_{num_class}classes_i8.tflite"
    tflite_model_i8_file.write_bytes(tflite_i8_model)
