'''https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/'''
# import
from scipy.spatial import distance as dist
from collections import OrderedDict 
import numpy as np

class CentroidTracker():

    # initialization of the class
    def __init__(self, maxDistance=20, maxDisappeared=50):
	    self.nextObjectID = 0
	    self.objects = OrderedDict()
	    self.disappeared = OrderedDict() #count number of frame object with that key is disappeared
	    self.maxDisappeared = maxDisappeared
	    self.maxDistance = maxDistance

    # to register a new object, to be sure to have a unique ID we use
    # the nextObjectID
    def register(self, centroid):
	    self.objects[self.nextObjectID] = centroid
	    self.disappeared[self.nextObjectID] = 0
	    self.nextObjectID += 1

    # eliminate object
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def count(self):
        return self.nextObjectID

    def update(self, rects):
		# check to see if the list of input bounding box rectangles
		# is empty
        if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
			# return, nothing to upload more
            return self.objects               
        # initialize a matrix with len(rects) rows and 2 column, so after will be x,y
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
		# loop over the bounding box rectangles
        for (i, (startX, startY, width, height)) in enumerate(rects):
            cX = int(startX+(width/2.0))
            cY = int(startY+(height/2.0))
            inputCentroids[i] = (cX,cY)
        # initialization if no object tracking
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
		# we try to match the input centroids to existing object
		# centroids
        else:
			# grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
			# compute the distance. D is a matrix
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value is at the *front* of the index
			# list
            rows = D.min(axis=1).argsort()
			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
			# loop over the combination of the (row, column) index
			# tuples
            for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue
                # if distance is too large ignore it 
                if D[row][col] > self.maxDistance:
                    continue
				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
				# indicate that we have examined each of the row and
				# column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
			# examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
            if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
                for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # there are new object to insert because they don't update
            # any existing centroid
            for col in unusedCols:
                self.register(inputCentroids[col])
		# return the set of trackable objects
        return self.objects