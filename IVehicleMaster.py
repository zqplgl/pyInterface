#coding=utf-8
from vehicleZoneDetect import IVehicleZoneDetect
from vehicleTypePosture import IVehiclePostureClassifier
from vehicleTypeColor import IVehicleColorClassifier
from vehicleTypeHead import IVehicleHeadClassifier
from vehicleTypeTail import IVehicleTailClassifier
from vehicleStruct import IVehicleStructure
from plateRecognize import IPlateRecognize

import os
import cv2
import time

class vehicleMaster:
    def __init__(self,modelDir,gpu_id,vehicleColor=True,vehicleType=True,vehicleStruct=True):
        self.__vehicleZoneDetector = IVehicleZoneDetect(modelDir,gpu_id)
        self.__gpu_id = gpu_id
        self.__vehicleColor = vehicleColor
        self.__vehicleType = vehicleType
        self.__vehicleStruct = vehicleStruct

        self.__vehiclePlate = IPlateRecognize(modelDir)

        if self.__vehicleColor:
            self.__vehicleColorClassifier = IVehicleColorClassifier(modelDir,gpu_id)

        if self.__vehicleType:
            self.__vehiclePostureClassifier = IVehiclePostureClassifier(modelDir,gpu_id)
            self.__vehicleHeadClassifier = IVehicleHeadClassifier(modelDir,gpu_id)
            self.__vehicleTailClassifier = IVehicleTailClassifier(modelDir,gpu_id)

        if self.__vehicleStruct:
            self.__vehicleStructDetector = IVehicleStructure(modelDir,gpu_id)

    def __getDetectStruct(self):
        result = {}
        result['vehicleZone'] = None
        result['vehicleColor'] = None
        result['vehiclePosture'] = None
        result['vehicleType'] = None
        result['vehiclePlateLicense'] = None
        result['vehicleStruct'] = None

        return result

    def detect(self,im):
        results = []
        zoneDetectResult = self.__vehicleZoneDetector.detect(im)

        for zone in zoneDetectResult:
            result = self.__getDetectStruct()
            result['vehicleZone'] = zone
            im_roi = im[zone[1]:zone[3],zone[0]:zone[2]]
            result['vehiclePlateLicense'] = self.__vehiclePlate.detect(im_roi.copy())

            if self.__vehicleColor:
                colorResult = self.__vehicleColorClassifier.classify(im_roi)
                result['vehicleColor'] = colorResult

            if self.__vehicleType:
                postureResult = self.__vehiclePostureClassifier.classify(im_roi)
                result['vehiclePosture'] = postureResult

                if postureResult['category']=="车头":
                    headResult = self.__vehicleHeadClassifier.classify(im_roi)
                    result['vehicleType'] = headResult

                    if self.__vehicleStruct:
                        structResult = self.__vehicleStructDetector.detect(im_roi)
                        result['vehicleStruct'] = structResult

                if postureResult['category']=="车尾":
                    tailResult = self.__vehicleTailClassifier.classify(im_roi)
                    result['vehicleType'] = tailResult

            results.append(result)

        return results

def run():
    modelDir = r'/home/zqp/install_lib/vehicleDll/models'
    master = vehicleMaster(modelDir,0,False,False,False)
    picDir = r'/media/zqp/新加卷/dataSet/data/carData5/data_1/'

    for picName in os.listdir(picDir):
        print picDir+picName
        if not picName.endswith(".jpg"):
            continue
        im = cv2.imread(picDir+picName)
        start = time.time()
        result = master.detect(im)
        print result
        end = time.time()

        print "cost time: ",(end-start)*1000,"ms","*********",len(result)

if __name__ == '__main__':
    run()
