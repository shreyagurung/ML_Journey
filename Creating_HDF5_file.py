# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 19:34:02 2019

@author: Shreya Gurung
"""

from random import shuffle
import glob
import numpy as np
import h5py
import cv2

shuffle_data = True

def HDF5file(store_hdf5_path, train_path, test_path):
    
    hdf5_path = store_hdf5_path //folder where you want to save your hdf5 model
    addrs_train = glob.glob(train_path)
    addrs_test = glob.glob(test_path)

    labels_train = [0 if 'cat' in addr else 1 for addr in addrs_train] 
    labels_test =  [0 if 'cat' in addr else 1 for addr in addrs_test]
    
    train_shape = (len(addrs_train), 224, 224, 3) //defining the size of the array
    test_shape = (len(addrs_test), 224, 224, 3) // defining the size of the array
    
    hdf5_file= h5py.File(hdf5_path, mode='w') //open your file in write mode
    
    hdf5_file.create_dataset("train_img", train_shape, np.int8)
    hdf5_file.create_dataset("test_img", test_shape, np.int8)
    
    hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)
    
    hdf5_file.create_dataset("train_labels", (len(addrs_train),), np.int8)
    hdf5_file["train_labels"][...] = labels_train
    hdf5_file.create_dataset("test_labels", (len(addrs_test),), np.int8)
    hdf5_file["test_labels"][...] = labels_test
    mean = np.zeros(train_shape[1:], np.float32)
    
    for i in range(len(addrs_train)):
        if i %1000 == 0 and i > 1:
            print ("TRAIN DATA: {}/{}".format(i, len(addrs_train)))
            
        addr = addrs_train[i]
        img = cv2.imread(addr)
        img = cv2.resize(img,(224,224), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        hdf5_file["train_img"][i, ...] = img[None]
        mean += img / float(len(labels_train))
        
    for i in range(len(addrs_test)):
        if i %1000 == 0 and i > 1:
            print ("TRAIN DATA: {}/{}".format(i, len(addrs_test)))
            
        addr = addrs_test[i]
        img = cv2.imread(addr)
        img = cv2.resize(img,(224,224), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        hdf5_file["test_img"][i, ...] = img[None]
        
    hdf5_file["train_mean"][...] = mean
    hdf5_file.close()
    
    
if __name__ == "__main__":
    HDF5file(hdf5_file_path, train_set_path,test_set_path)
    print("Sucessfully created the HDF5 file")
    
