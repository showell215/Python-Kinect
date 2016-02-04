import cv2
#import pykinect2.PyKinectRuntime as krt # (comment out kinect stuff)
#import pykinect2.PyKinectV2 as k2
import matplotlib.pyplot as plt
import numpy as np

def show(img, colormap='gray'):
    """ show an image (color infrared or depth) """
    t = plt.imshow(img, colormap)
    plt.show(t)

def load_uint16(name):
    """ load an uint16 image (infrared or depth) """
    data = None
    with open(name, 'r+b') as f:
        data = f.read()
    convData = np.empty(len(data)/2, np.uint16)
    count = 0
    for i in range(0,len(data),2):
        convData[count] = data[i] | (data[i+1] << 8)
        count += 1
    return convData

def load_bgra(name):
    """ load a color image (in bgra format) """
    with open(name, 'r+b') as f:
        data = f.read()
    convData = np.empty(len(data), np.uint8)
    count = 0
    for i in range(0,len(data),4):
        convData[count] = data[i+2] # red
        count += 1
        convData[count] = data[i+1] # green
        count += 1
        convData[count] = data[i] # blue
        count += 1
        convData[count] = data[i+3] # alpha
        count += 1
    return convData

def load(name):
    """ utility method that will load a color, depth, and infrared image all at once """
    color = load_bgra('{}.bgra'.format(name)).reshape((1080,1920,4))[:,:,0:3]
    infrared = load_uint16('{}.bgra.infrared'.format(name)).reshape((424,512))
    # for the depth below, we are subtracting from 65535 so the closer something is, the larger the number
    depth = 65535- (load_uint16('{}.bgra.depth'.format(name)).reshape((424,512)))
    return (color,depth,infrared)

def to_uint8(data):
    """ convert a uint16 image (depth or infrared) to uint8 because opencv likes uint8 images """
    min = np.min(data)
    max = np.max(data)
    normalData = (data-min)/(max-min)
    normalData = np.uint8(normalData * 255)
    return normalData

#class KinectHelper(object):
#    def __init__(self, frametypes=k2.FrameSourceTypes_Color|k2.FrameSourceTypes_Depth):
#        self.sensor = krt.PyKinectRuntime(frametypes)
#        if not self.sensor._sensor.IsAvailable:
#            raise RuntimeError('A kinect sensor was not found.')
#
#    def depth_space_to_color_space(self, x, y, depth):
#        depthPoint = k2._DepthSpacePoint(x, y)
#        colorPoint = self.sensor._mapper.MapDepthPointToColorSpace(depthPoint, depth)
#        return (round(colorPoint.x), round(colorPoint.y))