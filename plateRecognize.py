#coding=utf-8
import _vehicleDll as pr
import cv2
import os

class IPlateRecognize:
    def __init__(self,modelDir,gpu_id=0):
        self.__detector = pr.PlateDetector(modelDir)
        self.__gpu_id = gpu_id

    def detect(self,im):
        result =[]
        r = self.__detector.detect(im,im.shape[1],im.shape[0])
        for i in range(r.size()):
            temp = {} 
            temp["license"] = r[i].license
            temp['color'] = r[i].color
            temp['zone'] = (r[i].zone.x,r[i].zone.y,r[i].zone.x+r[i].zone.w,r[i].zone.y+r[i].zone.h)
            temp['score'] = r[i].score

            result.append(temp)

        sorted(result,key=lambda obj:obj['score'])

        if len(result):
            return result[0]
        else:
            return None


def run():
    modelDir = r"/home/zqp/install_lib/vehicleDll/models"
    plateDetector = IPlateRecognize(modelDir)
    picDir = "/media/zqp/新加卷/dataSet/data/vehicleHead/carHeadType_newType2014_2017/Jeep_Jeep进口_自由光进口_201420152016款_都市版/"

    for picName in os.listdir(picDir):
        
        im = cv2.imread(picDir+picName)
        result = plateDetector.detect(im)

        cv2.imshow("im",im)
        if result:
            print result['license'],"\t",result['color']

        if cv2.waitKey(0)==27:
            break


if __name__=="__main__":
    run()

