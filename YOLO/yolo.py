'''
Part of the project VEHICLE COUNTING
Cognitive services course, a.a. 2020-21
University of Padua
'''

import cv2 as cv
import numpy as np
import time

class YOLO:
    '''
    YOLO Class implementing yolo models thanks OpenCV library.
    '''

    def __init__(self, version='v3', verbose=1):
        '''
        Public constructor. Configuration and weights files are needed.
        For help check the link: https://github.com/AlexeyAB/darknet#pre-trained-models
        '''

        # Input shape and number of channels
        self.image_size = (416, 416)
        self.num_channels = 3

        # Loading the correct model with respect the version selected
        # Assumption: weights-files and cfg-files are trained for MS COCO dataset
        if version == 'v3':
            self.net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
        #elif version == 'v4':
        #    self.net = cv.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
        elif version == 'v2':
            self.net = cv.dnn.readNetFromDarknet('yolov2.cfg', 'yolov2.weights')
        else:
            print('Error loading the network')
            exit(-1)

        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)


        # If verbose is equal to 1, then printing the net configuration
        # if verbose is equal to 0, don't
        ln = self.net.getLayerNames()
        if verbose:
            print(len(ln))
            print()
            for elem in ln:
                print(elem)


    def predict(self, img, swapRB=True, crop=False, verbose=1, resize=1, threshold=0.5):
        '''
        Process a given image.
        '''

        if resize:
            img = cv.resize(img, self.image_size)

        # Output layer
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # Forward propagation
        blob = cv.dnn.blobFromImage(img, 1/255.0, self.image_size, swapRB=swapRB, crop=crop)
        self.net.setInput(blob)
        t0 = time.time()
        outputs = self.net.forward(ln)
        t = time.time()

        # If verbose is equal to 1, then printing computational time
        # if verbose is equal to 0, don't
        if verbose:
            print('time={}'.format(t-t0))



        # -------------- Boxes selection
        boxes = []
        confidences = []
        classIDs = []
        h, w = img.shape[:2]

        # 'Outputs' are of three types:
        # 1. large objects
        # 2. medium objects
        # 3. small objects
        for out in outputs:
            # For each type we have a different number of detections
            # Each detection is a vector of 85 elements
            # - 4 bounding boxe coordinates
            # - 1 bounding box score (probability that contains an object)
            # - 80 probabilities to have class 'i' of MS COCO dataset
            # In this application we are interested in having vehicle labels
            #    ids from 2 to 9
            for detection in out:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > threshold and (classID in range(2,10)):
                    # Original box format
                    # Center coordinates
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")

                    # New box format
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]

                    boxes.append(box)
                    confidences.append(float(confidence))
                    # The plus 1 is because the MS COCO dataset starts with ID equals 1
                    # and not zero 
                    classIDs.append(classID + 1)
                    
        if verbose:
            return [boxes, confidences, classIDs, t-t0]

        return [boxes, confidences, classIDs]

    def nms(self, boxes, confidences, threshold_score, threshold_overlap):
        idxs = cv.dnn.NMSBoxes(boxes, confidences, threshold_score, threshold_overlap)
        return idxs