import keras
import numpy as np
from plot_masks import plt_prediction
from training import dice_coef
from training import dice_coef_loss

with keras.utils.custom_object_scope({'dice_coef_loss': dice_coef_loss}):
    model = keras.models.load_model('../results/best_model_12_09_2024_15:34:29.keras')
    X_test = np.load('../dataset/seg/220/v3/X_test_128.npy')
    Y_test = np.load('../dataset/seg/220/v3/Y_test_128.npy')

    pred_test = [54, 42, 72, 0, 69]
    for test_img in pred_test:
        prediction = model.predict(X_test[test_img][np.newaxis, :, :, :])
        plt_prediction(X_test[test_img], Y_test[test_img], prediction, num_class=2)
