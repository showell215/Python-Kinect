import test #functions for loading .bgra images from kinect, converting between uint16/uint8
import cv2 #openCV
import matplotlib.pyplot as plt #for visualization
import numpy as np

"""
Takes 2 images captured from the Kinect sensor - one with the subject, one without,
and subtracts the background image from the image with subject for color, infrared,
and depth images. Returns the subtracted images in a tuple.
"""
def sub_BG(subject, noSubject):
   #load the color, depth, and IR img w/ subject
   c,d,i = test.load(subject)
   #load the color, depth, and IR img w/o subject
   cN,dN,iN = test.load(noSubject)

   #convert depth and IR to uint8 (openCV's preferred format)
   d = test.to_uint8(d)
   dN = test.to_uint8(dN)
   i = test.to_uint8(i)
   iN = test.to_uint8(iN)

   #subtract background from subject in depth image
   noBG_depth = cv2.subtract(d, dN)
   #perform a threshold operation to add contrast to subject in depth image
   ret, noBG_depth = cv2.threshold(noBG_depth, 1, 255, cv2.THRESH_BINARY)
   #subtract background from subject in IR image
   noBG_IR = cv2.subtract(i, iN)
   #subtract background from subject in color image
   noBG_color = cv2.subtract(c, cN)
   #perform a threshold operation to add contrast to subject in IR image
   #ret, noBG_IR = cv2.threshold(noBG_IR, 1, 255, cv2.THRESH_BINARY)

   return (noBG_color, noBG_depth, noBG_IR)
"""
Loads an image called 'name.bgra', captured from Kinect, and returns its color,
depth, and infrared components in a tuple.
"""
def load(name):
    """ utility method that will load a color, depth, and infrared image all at once """
    color = test.load_bgra('{}.bgra'.format(name)).reshape((1080,1920,4))[:,:,0:3]
    infrared = test.load_uint16('{}.bgra.infrared'.format(name)).reshape((424,512))
    # for the depth below, we are subtracting from 65535 so the closer something is, the larger the number
    depth = 65535- (test.load_uint16('{}.bgra.depth'.format(name)).reshape((424,512)))
    return (color,depth,infrared)


"""
TEST using images from 1 meter distance, 1 meter height, subject at Side Facing position
"""
c,d,i = sub_BG('1d1hSide', '1d1hNone')
test.show(c) #display the image until user closes
test.show(d) #display the image until user closes
test.show(i) #display the image until user closes
