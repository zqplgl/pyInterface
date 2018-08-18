#coding=utf-8
import os
import cv2
from objTypeClassifier import ObjTypeClassifier


class IVehiclePostureClassifier():
    def __init__(self,modelDir,gpu_id):
        modelDir += '/' if not modelDir.endswith('/') else modelDir

        assert(os.path.exists(modelDir+'vehiclePosture'))
        prototxt = modelDir+"vehiclePosture/vehiclePostureD3deploy.prototxt"
        assert(prototxt)
        weightfile = modelDir+'vehiclePosture/vehiclePostureD3model.dat'
        assert(weightfile)
        meanfile = modelDir+'vehiclePosture/vehiclePostureD3mean.dat'
        assert(meanfile)
        labelfile = modelDir+'vehiclePosture/vehiclePostureD3.txt'
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
    classifier = IVehiclePostureClassifier(modelDir,0)

    picDir = r'/media/zqp/新加卷/dataSet/data/vehicleHead/carHeadType/奥迪_一汽大众奥迪_奥迪100or红旗小红旗_200x款/'
    for picName in os.listdir(picDir):
        im = cv2.imread(picDir+picName)
        #im = caffe.io.load_image(picDir+picName)
        result = classifier.classify(im)
        print result['category'],"****************",result['score']
        if cv2.waitKey(0)==27:
            break

if __name__ == '__main__':
    run()

