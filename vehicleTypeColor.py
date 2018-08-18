#coding=utf-8
import os
import shutil
import cv2
import numpy as np
import math
import caffe
import pickle
import time
from objTypeClassifier import ObjTypeClassifier

class IVehicleColorClassifier():
    def __init__(self,modelDir,gpu_id):
        modelDir += '/' if not modelDir.endswith('/') else modelDir

        assert(os.path.exists(modelDir+'vehicleTypeColor'))
        prototxt = modelDir+"vehicleTypeColor/vehicleTypeColorD10deploy.prototxt"
        assert(prototxt)
        weightfile = modelDir+'vehicleTypeColor/vehicleTypeColorD10model.dat'
        assert(weightfile)
        meanfile = modelDir+'vehicleTypeColor/vehicleTypeColorD10mean.dat'
        assert(meanfile)
        labelfile = modelDir+'vehicleTypeColor/vehicleTypeColorID10.txt'
        assert(labelfile)

        self.__gpu_id = gpu_id
        self.__classifier = ObjTypeClassifier(prototxt,weightfile,meanfile,gpu_id)
        self.__labelMap = [line.strip() for line in open(labelfile).readlines()]

    def classify(self,im):
        result = {}

        info = self.__classifier.classify(im,'fc7')
        result['category'] = self.__labelMap[info[0]]
        result['score'] = info[1]

        return result

def run():
    modelDir = r'/home/zqp/install_lib/vehicleDll/models'
    classifier = IVehicleColorClassifier(modelDir,0)

    picDir = r'/media/zqp/新加卷/dataSet/data/vehicleHead/carHeadType/奥迪_一汽大众奥迪_奥迪100or红旗小红旗_200x款/'
    for picName in os.listdir(picDir):
        im = cv2.imread(picDir+picName)
        result = classifier.classify(im)
        print result['category'],"****************",result['score']
        cv2.imshow('im',im)
        if cv2.waitKey(0)==27:
            break

if __name__ == '__main__':
    run()

