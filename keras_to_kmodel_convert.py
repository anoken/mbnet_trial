## Copyright (c) 2019 aNoken


import keras
import numpy as np
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model_file('weight2.h5')
tflite_model = converter.convert()
open('my_mbnet.tflite', "wb").write(tflite_model)

import subprocess
subprocess.run(['./ncc/ncc','my_mbnet.tflite','my_mbnet.kmodel','-i','tflite','-o',
'k210model','--dataset','images'])




