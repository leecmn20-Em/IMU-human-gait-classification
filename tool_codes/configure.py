import os

os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf

def initiate() -> None:    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)