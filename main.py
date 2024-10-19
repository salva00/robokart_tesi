import numpy as np

from dataset_stat import get_stat
from preprocess import preprocess
from quantization import to_tf_lite
from training import train
#from test import test
if __name__ == '__main__':
    fov = 220
    version = 3
    size = (128, 128)
    basepath = "dataset/seg/" + f"{fov}" + "/v" + f"{version}/"
    preprocess(size=size, basepath=basepath, num_class=2)

    get_stat(basepath=basepath, size=size)
    #add
    model, model_name = train(size=size, num_class=2, basepath=basepath)
    test(model, bw=True)
    to_tf_lite(model, model_name, size=size, num_class=2, basepath=basepath)
    model, model_name = train(size=(64, 64), num_class=3, bw=True)