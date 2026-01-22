### CONTRIBUTOR: THẮNG

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from char_classification import config
from char_classification.data_provider import Datasets


ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}


class KNN_Model(object):
    def __init__(self, trainable=True):
        self.trainable = trainable
        self.n_neighbors = config.KNN_NEIGHBORS  # Add this to your config.py
        
        # Building model
        self._build_model()

        # Input data
        self.data = Datasets()

    def _build_model(self):
        # KNN model
        self.model = KNeighborsClassifier(n_neighbors = self.n_neighbors, weights='distance', metric='euclidean')
        
    def train(self):
        print("Training KNN model......")
        print(config.KNN_NEIGHBORS)
        trainX, trainY = self.data.gen()
        
        # Convert one-hot encoded labels back to categorical
        trainY = np.argmax(trainY, axis=1)
        
        # Reshape images from (N, 28, 28, 1) to (N, 784) for KNN
        trainX = np.array(trainX).reshape(len(trainX), -1)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            trainX, trainY, test_size=0.15, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_pred = self.model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        print(f"Validation Accuracy: {val_acc:.4f}")
        
    def predict(self, images):
        """Predict labels for input images"""

        if len(images.shape) == 3:  # If single image (28, 28, 1)
            images = images.reshape(1, -1)
        elif len(images.shape) == 4:  # If batch of images (N, 28, 28, 1)
            images = images.reshape(len(images), -1)
            
        predictions = self.model.predict(images)
        return [ALPHA_DICT[p] for p in predictions]
    
model = KNN_Model(trainable=True)    
model.train()