
# coding: utf-8

# In[1]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import warnings
warnings.filterwarnings("ignore")
marks = pd.read_csv('D:\Kaggle/train_ship_segmentations.csv') # Markers for ships
images = os.listdir('D:\Kaggle/train_images') # Images for training
os.chdir("D:\Kaggle/train_images")

## extra imports to set GPU options
import tensorflow as tf
from keras import backend as k
 
###################################
# TensorFlow wizardry
config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.9
 
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))
###################################

# In[2]:


# test = pd.read_csv('../COMPETITION/test_ship_segmentations.csv')
# test_images = os.listdir('../COMPETITION/test_images')


# In[3]:


# test["ImageId"].unique().shape[0] == len(test_images)
# len(test_images)


# In[4]:


marks['ships'] = marks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
# unique_img_ids = marks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()


# In[5]:


# marks.head()


# In[6]:


train_data = marks[marks["ships"]!=0]
train_ids = list(train_data["ImageId"])
no_ships_train = list(set(images) & set(train_ids))
train_images_ids = list(set(no_ships_train) & set(images))


# In[7]:


# train_data.shape


# In[8]:


def mask_part(pic):
    '''
    Function that encodes mask for single ship from .csv entry into numpy matrix
    '''
    back = np.zeros(768**2)
    starts = pic.split()[0::2]
    lens = pic.split()[1::2]
    for i in range(len(lens)):
        back[(int(starts[i])-1):(int(starts[i])-1+int(lens[i]))] = 1
    return np.reshape(back, (768, 768, 1))

def is_empty(key):
    '''
    Function that checks if there is a ship in image
    '''
    df = marks[marks['ImageId'] == key].iloc[:,1]
    if len(df) == 1 and type(df.iloc[0]) != str and np.isnan(df.iloc[0]):
        return True
    else:
        return False
    
def masks_all(key):
    '''
    Merges together all the ship markers corresponding to a single image
    '''
    df = marks[marks['ImageId'] == key].iloc[:,1]
    masks= np.zeros((768,768,1))
    if is_empty(key):
        return masks
    else:
        for i in range(len(df)):
            masks += mask_part(df.iloc[i])
        return np.transpose(masks, (1,0,2))


# In[9]:


def transform(X, Y):
    '''
    Function for augmenting images. 
    It takes original image and corresponding mask and performs the
    same flipping and rotation transforamtions on both in order to 
    perserve the overlapping of ships and their masks
    '''
# add noise:
    
    x = np.copy(X)
    y = np.copy(Y)

    x[:,:,0] = x[:,:,0] + np.random.normal(loc=0.0, scale=0.01, size=(768,768))
    x[:,:,1] = x[:,:,1] + np.random.normal(loc=0.0, scale=0.01, size=(768,768))
    x[:,:,2] = x[:,:,2] + np.random.normal(loc=0.0, scale=0.01, size=(768,768))
    # Adding Gaussian noise on each rgb channel; this way we will NEVER get two completely same images.
    # Note that this transformation is not performed on Y 
    x[np.where(x<0)] = 0
    x[np.where(x>1)] = 1
# axes swap:
    if np.random.rand()<0.5: # 0.5 chances for this transformation to occur (same for two below)
        x = np.swapaxes(x, 0,1)
        y = np.swapaxes(y, 0,1)
# vertical flip:
    if np.random.rand()<0.5:
        x = np.flip(x, 0)
        y = np.flip(y, 0)
# horizontal flip:
    if np.random.rand()<0.5:
        x = np.flip(x, 1)
        y = np.flip(y, 1)
    return x, y  


# In[10]:


def ship_gen(files):
    for id in files:
        yield id


# In[11]:


def make_batch(files, batch_size):
    '''
    Creates batches of images and masks in order to feed them to NN
    '''
    X = np.zeros((batch_size, 768, 768, 3))
    Y = np.zeros((batch_size, 768, 768, 1)) # I add 1 here to get 4D batch
    for i in range(batch_size):
        gen = ship_gen(files)
        ship = next(gen)
#         ship = np.random.choice(files)
        X[i] = (io.imread(ship))/255.0 # Original images are in 0-255 range, I want it in 0-1
        Y[i]= masks_all(ship)
    return X, Y


# In[12]:


def Generator(files, batch_size):
    '''
    Generates batches of images and corresponding masks
    '''
    while True:
        X, Y = make_batch(files, batch_size)
        for i in range(batch_size):
            X[i], Y[i] = transform(X[i], Y[i])
        yield X, Y


# In[13]:


# Intersection over Union for Objects
def IoU(y_true, y_pred, tresh=1e-10):
    Intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    Union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - Intersection
    return K.mean( (Intersection + tresh) / (Union + tresh), axis=0)
# Intersection over Union for Background
def back_IoU(y_true, y_pred):
    return IoU(1-y_true, 1-y_pred)
# Loss function
def IoU_loss(in_gt, in_pred):
    #return 2 - back_IoU(in_gt, in_pred) - IoU(in_gt, in_pred)
    return 1 - IoU(in_gt, in_pred)


# In[14]:


inputs = Input((768, 768, 3))

c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)

u5 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c4)
u5 = concatenate([u5, c3])
c5 = Conv2D(32, (3, 3), activation='relu', padding='same') (u5)
c5 = Conv2D(32, (3, 3), activation='relu', padding='same') (c5)

u6 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c2])
c6 = Conv2D(16, (3, 3), activation='relu', padding='same') (u6)
c6 = Conv2D(16, (3, 3), activation='relu', padding='same') (c6)

u7 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c1], axis=3)
c7 = Conv2D(8, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(8, (3, 3), activation='relu', padding='same') (c7)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c7)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer="adam", loss= IoU_loss, metrics=[IoU, back_IoU])
model.summary()


# In[15]:


import gc; gc.enable()
gc.collect()


# In[16]:


batch_size = 3
steps_per_epoch = int(train_data.shape[0]/(batch_size))+1
epochs = 1
print(steps_per_epoch)


# In[17]:


results = model.fit_generator(Generator(train_images_ids, batch_size = batch_size), steps_per_epoch = steps_per_epoch, epochs = epochs)


# In[ ]:


# model.save_weights('Unet_1epoch_1.h5')


# In[ ]:


# valid_df["ImageId"]


# In[ ]:


# a = io.imread(valid_df["ImageId"][2]).reshape(1,768,768,3)
# b = model.predict(a)


# In[ ]:


# b[b==1]


# # In[ ]:


# valid_df["ships"].max()


# # In[ ]:


# def foo(model, data, data_size = 10, epochs_for_data = 1, EPOCHS = 1, batch_size_for_data = 2):
#     lim = int(len(data)/data_size)
#     for i in range(EPOCHS):
#         for j in range(lim):
# #             print(1)
#             model.load_weights('Unet_weights_{0}.h5'.format(j))
# #             print(2)
#             current_data = data[j*data_size:(j+1)*data_size + 1]
# #             print(4)
#             results = model.fit_generator(Generator(current_data, batch_size = batch_size_for_data ), steps_per_epoch = 50, epochs = epochs_for_data)
# #             print(5)
#             model.save_weights('Unet_weights_{0}.h5'.format(j+1))
# #             print(6)
#             K.clear_session()
# #             print(7)
#     return model

