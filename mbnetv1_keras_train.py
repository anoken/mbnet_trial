## Copyright (c) 2019 aNoken
## https://anoken.jimdo.com/
## https://github.com/anoken/purin_wo_motto_mimamoru_gijutsu

import keras,os
import numpy as np
from keras import backend as K, Sequential
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.mobilenet import preprocess_input
import tensorflow as tf
from mobilenet_sipeed.mobilenet import MobileNet

NUM_CLASSES = 3
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
TRAINING_DIR = 'dataset/train'
VALIDATION_DIR = 'dataset/train'

imageGen=ImageDataGenerator(preprocessing_function=preprocess_input,validation_split = 0.1)
batch_size=5

train_generator=imageGen.flow_from_directory(TRAINING_DIR,
	target_size=(IMAGE_WIDTH,IMAGE_HEIGHT),color_mode='rgb',
	batch_size=batch_size,class_mode='categorical', shuffle=True, subset = "training")

validation_generator=imageGen.flow_from_directory(VALIDATION_DIR,
	target_size=(IMAGE_WIDTH,IMAGE_HEIGHT),color_mode='rgb',
	batch_size=batch_size,class_mode='categorical', shuffle=True,subset = "validation")


base_model=MobileNet(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), alpha = 0.5,depth_multiplier = 1,
 dropout = 0.001,include_top = False, weights = "imagenet", classes = 1000, backend=keras.backend, 
 layers=keras.layers,models=keras.models,utils=keras.utils)

# Additional Layers

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(100,activation='relu')(x)#
x=Dropout(0.5)(x)
x=Dense(50, activation='relu')(x)
preds=Dense(NUM_CLASSES, activation='softmax')(x)

mbnetModel=Model(inputs=base_model.input,outputs=preds)

for i,layer in enumerate(mbnetModel.layers):
    print(i,layer.name)

for layer in base_model.layers:
    layer.trainable = False

mbnetModel.summary()


mbnetModel.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


step_size_train = (train_generator.n//train_generator.batch_size)
validation_steps = (validation_generator.n//train_generator.batch_size)


class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch, logs=None):
        "各エポック終了時に重みを保存する"
        mbnetModel.save("weight.h5")

cb = Callback()
# 途中から学習する場合
initial_epoch = 0

if os.path.isfile(os.path.join("weight.h5")):    
    mbnetModel.load_weights(os.path.join("weight.h5"))


history=mbnetModel.fit_generator(generator=train_generator, 
	steps_per_epoch=step_size_train, epochs=100, 
	validation_data = validation_generator,validation_steps = validation_steps, verbose = 1,callbacks=[cb])

mbnetModel.save('my_mbnet.h5')

np.savetxt("model_top_loss.csv", history.history['loss'])
np.savetxt("model_top_val_loss.csv", history.history['val_loss'])
np.savetxt("model_top_acc.csv", history.history['acc'])
np.savetxt("model_top_val_acc.csv", history.history['val_acc'])
