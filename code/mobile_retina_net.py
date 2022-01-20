import sys
sys.path.insert(0, './utils')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
import pandas as pd
import models
import func
import load_images
from datetime import datetime
import numpy as np
import cv2 as cv
########################################################################################################################
np.random.seed(1)
epochs = 500
batch_size = 8

img_size=256
print('Loading data...')
X_train, y_train = load_images.load_images(img_size,gray=0)
print('Data loading complete...')
print(y_train.max())
########################################################################################################################
model_mob_vessel_net=models.mobile_vesselNet(img_dim=img_size)
print(model_mob_vessel_net.summary())
model_mob_vessel_net.compile(optimizer=Adam(learning_rate=0.0001),
                   loss=func.dice_coef_loss,
                   metrics=['acc',func.dice_coef])
earlystop=EarlyStopping(monitor='val_dice_coef',patience=50,mode='max')
checkpoint_path='results_patches/model_checkpoint_acc_mob_vessel_net_dice/Checkpoint_best'
log_dir = 'logs_patches/mob_vessel_net_dice' + datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint=ModelCheckpoint(filepath=checkpoint_path,
                           monitor='val_dice_coef',
                           save_best_only=True,
                           save_weights_only=True,
                           mode='max')

history_mob_vessel_net=model_mob_vessel_net.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,
                            verbose=1,validation_split=0.2,
                            callbacks=[earlystop,checkpoint,tensorboard_callback])
########################################################################################################################
history_mob_vessel_net_df=pd.DataFrame(history_mob_vessel_net.history)
history_mob_vessel_net_df.to_csv('results_patches/model_mob_vessel_net_dice.csv')
########################################################################################################################
# Performance measure