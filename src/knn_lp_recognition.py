# CONTRIBUTORS: THẮNG, BÌNH, HOÀNG ANH, BẢO
import cv2
import numpy as np
from skimage import measure
from imutils import perspective
import imutils

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .data_utils import order_points, convert2Square, draw_labels_and_boxes
from .lp_detection.detect import detectNumberPlate
from .char_classification.knn_model import KNN_Model
from skimage.filters import threshold_local

ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}
ok = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}


LP_DETECTION_CFG = {
    "weight_path": "./src/weights/yolov3-tiny_15000.weights",
    "classes_path": "./src/lp_detection/cfg/yolo.names",
    "config_path": "./src/lp_detection/cfg/yolov3-tiny.cfg"
}

class knn_E2E(object):
    def __init__(self):
        self.image = np.empty((28, 28, 1))
        self.detectLP = detectNumberPlate(LP_DETECTION_CFG['classes_path'], 
                                        LP_DETECTION_CFG['config_path'], 
                                        LP_DETECTION_CFG['weight_path'])
        # Initialize KNN model
        self.recogChar = KNN_Model(trainable=False)
        self.recogChar.train()
        self.candidates = []

    def extractLP(self):
        coordinates = self.detectLP.detect(self.image)
        if len(coordinates) == 0:
            ValueError('No images detected')

        for coordinate in coordinates:
            yield coordinate

### hoàng anh ----
    def predict(self, image):
        self.image = image

        for coordinate in self.extractLP():
            self.candidates = []
            pts = order_points(coordinate)
            LpRegion = perspective.four_point_transform(self.image, pts)
           
            self.segmentation(LpRegion)
            self.recognizeChar()
            license_plate = self.format()
            self.image = draw_labels_and_boxes(self.image, license_plate, coordinate)

        return self.image
# bình---
    def segmentation(self, LpRegion):

        V = cv2.split(cv2.cvtColor(LpRegion, cv2.COLOR_BGR2HSV))[2]
        T = threshold_local(V, 15, offset=10, method="gaussian")
        thresh = (V > T).astype("uint8") * 255
        thresh = cv2.bitwise_not(thresh)
        thresh = imutils.resize(thresh, width=400)
        thresh = cv2.medianBlur(thresh, 5)

        labels = measure.label(thresh, connectivity=2, background=0)

        for label in np.unique(labels):
            if label == 0:
                continue

            mask = np.zeros(thresh.shape, dtype="uint8")
            mask[labels == label] = 255

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(contour)

                aspectRatio = w / float(h)
                solidity = cv2.contourArea(contour) / float(w * h)
                heightRatio = h / float(LpRegion.shape[0])

                if 0.1 < aspectRatio < 1.0 and solidity > 0.1 and 0.35 < heightRatio < 2.0:
                    candidate = np.array(mask[y:y + h, x:x + w])
                    square_candidate = convert2Square(candidate)
                    square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)
                    square_candidate = square_candidate.reshape((28, 28, 1))
                    self.candidates.append((square_candidate, (y, x)))
# Bảo
    def recognizeChar(self):
        characters = []
        coordinates = []
        
        for char, coordinate in self.candidates:
            # Flatten image for KNN (28,28,1) -> (784,)
            char_flat = char.reshape(1, -1)
            characters.append(char)  # Remove batch dimension
            coordinates.append(coordinate)
            

        if characters:
            
            
            characters = np.array(characters)
            # KNN returns class indices directly
            result_idx = self.recogChar.predict(characters)
            
            self.candidates = []
            for i in range(len(result_idx)):
                if result_idx[i] == '"Background"':  # Background class
                    continue
                
                self.candidates.append((result_idx[i], coordinates[i]))

# thắng --
    def format(self):
        first_line = []
        second_line = []

        for candidate, coordinate in self.candidates:
            if self.candidates[0][1][0] + 40 > coordinate[0]:
                first_line.append((candidate, coordinate[1]))
            else:
                second_line.append((candidate, coordinate[1]))

        def take_second(s):
            return s[1]

        first_line = sorted(first_line, key=take_second)
        second_line = sorted(second_line, key=take_second)
        chars = [ele[0] for ele in first_line + second_line]
        replace = {'D': '0', 'T': '1', 'S': '5','P':'8','Z':'2'}
        new_chars = []
        for i, char in enumerate(chars):
            if char in replace:
                new_chars.append(replace[char])
            else:
                new_chars.append(char)



        if len(second_line) == 0:
            license_plate = "".join(new_chars)
        else:
            license_plate = "".join([str(ele[0]) for ele in first_line]) + "-" + \
                          "".join([str(ele[0]) for ele in second_line])

        return license_plate
### thắng --
    def get_license_plate_list(self):
            if not hasattr(self, 'candidates') or not self.candidates:
                return None
                
            first_line = []
            second_line = []

            for candidate, coordinate in self.candidates:
                if self.candidates[0][1][0] + 40 > coordinate[0]:
                    first_line.append((candidate, coordinate[1]))
                else:
                    second_line.append((candidate, coordinate[1]))

            def take_second(s):
                return s[1]

            first_line = sorted(first_line, key=take_second)
            second_line = sorted(second_line, key=take_second)

            if len(second_line) == 0:  # 1 line
                return [[char[0] for char in first_line]]
            else:   # 2 lines (không cần )
                return [
                    [char[0] for char in first_line],
                    [char[0] for char in second_line]
                ]        