#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import os

for dirname, _, filenames in os.walk(r'F:\Tejaswini\6th_Sem\DL\Game of deep learning\train\images'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Importing Necessary Libraries

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')


# # Data Import

# In[3]:


train = pd.read_csv(r'F:\Tejaswini\6th_Sem\DL\Game of deep learning\train\train.csv')


# # Data Analysis

# In[4]:


train.head()


# In[5]:


train.tail()


# In[6]:


train.describe()


# In[7]:


train.info()


# In[8]:


train.shape


# In[9]:


train.isnull().sum()


# In[10]:


ship_categories = {1: 'Cargo', 2: 'Military', 3: 'Carrier', 4: 'Cruise', 5: 'Tanker'}
sns.countplot(x=train["category"].map(ship_categories))


# In[11]:


pie = train.loc[:, "category"].value_counts()
pie.plot.pie(autopct='%.2f')


# # Data Preprocessing

# **Function to load and preprocess an image**

# In[12]:


path = r'F:\Tejaswini\6th_Sem\DL\Game of Deep Learning\train\images'
target_shape = (128, 128, 3)

def load_and_preprocess_image(image_path, target_shape):
    img = plt.imread(image_path)
    img = cv2.resize(img, (target_shape[1], target_shape[0]))  # Resize the image
    return img


# **Load and preprocess images to an array of RGB colors**

# In[13]:


refactor_size = 128
resized_image_list = []
all_paths = []

for i in range(len(train)):
    image_path = os.path.join(path, train["image"][i])
    img = tf.keras.utils.load_img(image_path, target_size=(refactor_size, refactor_size))
    img_vals = tf.image.convert_image_dtype(img, tf.float32)
    imgarr = tf.keras.utils.img_to_array(img_vals)
    resized_image_list.append(imgarr)
    all_paths.append(image_path)
    
resized_image_list = np.asarray(resized_image_list) #List of preprocessed images


# **Plotting first 20 Images**

# In[14]:


nrow = 5
ncol = 4
fig1 = plt.figure(figsize=(15, 15))
fig1.suptitle('After Resizing', size=25)
for i in range(20):
    plt.subplot(nrow, ncol, i + 1)
    plt.imshow(resized_image_list[i])
    plt.title('class = {x}, Ship is {y}'.format(x=train["category"][i], y=ship_categories[train["category"][i]]))
    plt.axis('Off')
    plt.grid(False)
plt.show()


# In[15]:


# Saving the preprocessed images in the desired folder

# from PIL import Image
# import csv

# output_dir = r'Game of deep learning\DL\train\Preprocessed'

# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
    
# for i, img_arr in enumerate(resized_image_list):
#     img_pil = Image.fromarray(np.uint8(img_arr*255))
#     img_path = os.path.join(output_dir, f'resized_image_list_{i}.jpg')
#     img_pil.save(img_path)
#     print(f'Saved images {i} to {img_path}')


# In[16]:


# !pip install opencv-python


# #### Data augmentation 

# In[21]:


data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal'), tf.keras.layers.RandomRotation(0.2),])
augmented_images = data_augmentation(resized_image_list)


# In[18]:


fig2 = plt.figure(figsize=(15, 15))
fig2.suptitle('After Augmentation', size=32)
for i in range(20):
    plt.subplot(nrow, ncol, i + 1)
    plt.imshow(augmented_images[i])
    plt.title('class = {x}, Ship is {y}'.format(x=train["category"][i], y=ship_categories[train["category"][i]]))
    plt.axis('Off')
    plt.grid(False)
plt.show()


# In[19]:


cat_values = train["category"] - 1
cat_values.value_counts()


# In[20]:


ship_categories = {0: 'Cargo', 1: 'Military', 2: 'Carrier', 3: 'Cruise', 4: 'Tanker'}


# In[ ]:




