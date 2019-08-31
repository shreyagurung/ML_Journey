# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 19:40:43 2019

@author: Shreya Gurung
"""

from keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat
        
    def preprocess(self, image):
        return img_to_array(image, data_format=self.dataFormat)
