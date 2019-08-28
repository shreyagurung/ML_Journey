import os, cv2, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import scipy
from scipy import ndimage

import matplotlib.pyplot as plt

def load_dataset(train_path,test_path):
  train_dir = train_path
  test_dir = test_path
  train_images = [train_dir+i for i in os.listdir(train_dir)]
  test_images = [test_dir+i for i in os.listdir(test_dir)]
  X_train, y_train = prep_data(train_images)
  X_test, test_idx = prep_data(test_images)
  classes = {0: 'cats',
          1:'dogs'}
  return X_train,y_train,X_test,test_idx,classes

  rows = 64
  cols = 64
  channels = 3
  
def read_image(file_path):
    img = cv2.imread(file_path)
    if img is not None:
        resized_img = cv2.resize(img, (rows, cols), interpolation = cv2.INTER_CUBIC)
        return resized_img
    else:
        print("Image not loaded")
        
def prep_data(images):
    m = len(images)
    n_x = rows * cols *channels
    
    X = np.ndarray((n_x, m), dtype=np.uint8)
    y=np.zeros((1,m))
    print("X.shape is {}".format(X.shape))

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        X[:,i] = np.squeeze(image.reshape((n_x, 1)))
        if 'dog' in image_file.lower() :
            y[0, i] = 1
        elif 'cat' in image_file.lower():
            y[0, i] = 0
        else:
            y[0, i] = imagefile.split('/')[-1].split('.')[0]
            
        if i%5000 == 0:
            print("Proceed {} of {}".format(i, m))
        
    return X, y
