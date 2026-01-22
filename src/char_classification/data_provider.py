import tensorflow.keras as keras
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Giả định rằng data_utils.py tồn tại và có các hàm get_digits_data, get_alphas_data
from data_utils import get_digits_data, get_alphas_data

class Datasets(object):
    def __init__(self):
        self.all_data = []

        # Input data
        # hàm get_digits_data có đầu vào là path của file npynpy
        #C:\Users\Nguyen Van Thang\Documents\GitHub\License-Plate-Recognition\data\digits.npynpy
        
        self.digits_data = get_digits_data(r'C:\Users\Nguyen Van Thang\Documents\GitHub\License-Plate-Recognition\data\digits.npy')
        self.alphas_data = get_alphas_data(r'C:\Users\Nguyen Van Thang\Documents\GitHub\License-Plate-Recognition\data\alphas.npy')

        # Preprocess
        self.convert_data_format()

    def gen(self):
        np.random.shuffle(self.all_data)
        images = []
        labels = []

        for i in range(len(self.all_data)):
            image, label = self.all_data[i]
            images.append(image)
            labels.append(label)

        labels = keras.utils.to_categorical(labels, num_classes=32)
        return np.array(images), np.array(labels)  # Chuyển đổi sang numpy array

    def convert_data_format(self):
        # Digits data
        for i in range(len(self.digits_data)):
            image = self.digits_data[i][0]
            label = self.digits_data[i][1]
            self.all_data.append((image, label))

        # Alpha data
        nb_alphas_data = len(self.alphas_data)
        for i in range(nb_alphas_data * 8):
            image = self.alphas_data[i % nb_alphas_data][0]
            label = self.alphas_data[i % nb_alphas_data][1]
            self.all_data.append((image, label))
