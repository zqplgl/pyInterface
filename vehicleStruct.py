#coding=utf-8
import cv2
import os
from objZoneDetectFaster import ObjZoneDetectFaster

class IVehicleStructure:
    def __init__(self,modelDir,gpu_id):
        modelDir += '/' if not modelDir.endswith('/') else modelDir
        assert(os.path.exists(modelDir+"vehicleStructure"))
        prototxtD3 = modelDir+r'vehicleStructure/vehicleStructD3deploy.prototxt'
        assert(prototxtD3)
        weightfileD3 = modelDir+r'vehicleStructure/vehicleStructD3model.dat'
        assert(weightfileD3)
        prototxtD10 = modelDir+r'vehicleStructure/vehicleStructD10deploy.prototxt'
        assert(prototxtD10)
        weightfileD10 = modelDir+r'vehicleStructure/vehicleStructD10model.dat'
        assert(weightfileD10)

        self.__gpu_id = gpu_id
        self.__confidence = 0.65
        self.__detectorD3 = ObjZoneDetectFaster(prototxtD3,weightfileD3,gpu_id,600,1000,self.__confidence)
        self.__detectorD10 = ObjZoneDetectFaster(prototxtD10,weightfileD10,gpu_id,600,1000,self.__confidence)

    def __getOutPutStruct(self):
        result = {}
        result['withSkyRoof'] = False
        result['isTaxi'] = False
        result['withDriver'] = False
        result['withOtherPeopleOnSideSeat'] = False
        result['withSideSafetyBelt'] = False
        result['withSunShieldDown'] = False
        result['withDriverSafetyBelt'] = False
        result['withFrontWindowObjects'] = False
        result['withFrontWindowAccessories'] = False
        result['withFrontWindowLabelInspection'] = False
        result['withCellPhone'] = False

        return result

    def __writeVehicleResult(self,resultD3,result):
        frontGlass = False
        info = None
        for label in resultD3.keys():
            if label==1:
                result['withSkyRoof'] = True
                result['skyRoof'] = sorted(resultD3[label],key=lambda obj:obj[-1])[-1]
            elif label==3:
                result['taxi'] = sorted(resultD3[label],key=lambda obj:obj[-1])[-1]
                result['isTaxi'] = True
            else:
                info = sorted(resultD3[label],key=lambda obj:obj[-1])[-1]
                frontGlass = True

        return frontGlass,info

    def __writeFrontGlassResult(self,resultD10,result):
        driverSeatZone = [] 
        labelSeatZone = []
        for label in resultD10.keys():
            if label in [1,2,9,10]:
                result['withDriver'] = True
                driverSeatZone.append(sorted(resultD10[label],key=lambda obj:obj[-1])[-1])
                labelSeatZone.append(label)
            elif label in [3,4]:
                result['withOtherPeopleOnSideSeat'] = True

                if not result.has_key('sideSeatZone'):
                    result['sideSeatZone'] = []
                result['sideSeatZone'].extend(resultD10[label])

                if label==4:
                    result['withSideSafetyBelt'] = True

            elif label==5:
                result['withSunShieldDown'] = True
                result['frontWindowSunShield'] = resultD10[label]
            elif label==6:
                result['withFrontWindowObjects'] = True
                result['frontWindowObjectsZone'] = resultD10[label]
            elif label==7:
                result['withFrontWindowAccessories'] = True
                result['frontWindowAccessoriesZone'] = resultD10[label]
            elif label==8:
                result['withFrontWindowLabelInspection'] = True
                result['frontWindowLabelInspectionZone'] = resultD10[label]

        score = 0
        index = 0
        for i in range(len(driverSeatZone)):
            if score<driverSeatZone[i][-1]:
                score = driverSeatZone[i][-1]
                index = i

        if score!=0:
            result['driveSeatZone'] = driverSeatZone[index]
            if labelSeatZone[index] in [2,9]:
                result['withDriverSafetyBelt'] = True
            elif labelSeatZone[index] in [9,10]:
                result['withCellPhone'] = True

        pass

    def detect(self,im):
        result = self.__getOutPutStruct()
        resultD3 = self.__detectorD3.detect(im)

        frontGlass,info = self.__writeVehicleResult(resultD3,result)

        if not frontGlass:
            return result
        imFrontGlass = im[info[1]:info[3],info[0]:info[2]]
        #cv2.imshow('imFrontGlass',imFrontGlass)
        resultD10 = self.__detectorD10.detect(imFrontGlass)
        for key in resultD10.keys():
            for zone in resultD10[key]:
                zone[0] += info[0]
                zone[1] += info[1]
                zone[2] += info[0]
                zone[3] += info[1]

        self.__writeFrontGlassResult(resultD10,result)

        return result

def addRectangle(im,result):
   #天窗
    if result['withSkyRoof']:
        info = result['skyRoof']
        cv2.rectangle(im,(info[0],info[1]),(info[2],info[3]),(0,0,255),2)

    #出租车
    if result['isTaxi']:
        info = result['taxi']
        cv2.rectangle(im,(info[0],info[1]),(info[2],info[3]),(0,0,255),2)

    #副驾驶
    if result['withOtherPeopleOnSideSeat']:
        for info in result['sideSeatZone']:
            cv2.rectangle(im,(info[0],info[1]),(info[2],info[3]),(0,0,255),2)
    #副驾驶是否系安全带
    print  "SideSafetyBelt: ",result['withSideSafetyBelt']

    #遮阳板
    if result['withSunShieldDown']:
        for info in result['frontWindowSunShield']:
            cv2.rectangle(im,(info[0],info[1]),(info[2],info[3]),(0,0,255),2)

    #驾驶员
    if result['withDriver']:
        info = result['driveSeatZone']
        cv2.rectangle(im,(info[0],info[1]),(info[2],info[3]),(0,0,255),2)
    #驾驶员是否系安全带
    print "withDriverSafetyBelt: ",result['withDriverSafetyBelt']
    print "withCellPhone: ", result['withCellPhone']

    #摆件
    if result['withFrontWindowObjects']:
        for info in result['frontWindowObjectsZone']:
            cv2.rectangle(im,(info[0],info[1]),(info[2],info[3]),(0,0,255),2)
    #挂件
    if result['withFrontWindowAccessories']:
        for info in result['frontWindowAccessoriesZone']:
            cv2.rectangle(im,(info[0],info[1]),(info[2],info[3]),(0,0,255),2)
    #标签
    if result['withFrontWindowLabelInspection']:
        for info in result['frontWindowLabelInspectionZone']:
            cv2.rectangle(im,(info[0],info[1]),(info[2],info[3]),(0,0,255),2)

def run():
    modelDir=r'/home/zqp/install_lib/vehicleDll/models'
    gpu_id = 0
    detector = IVehicleStructure(modelDir,gpu_id)

    picDir = r'/media/zqp/新加卷/dataSet/data/vehicleHead/carHeadType/奥迪_一汽大众奥迪_奥迪100or红旗小红旗_200x款/'
    for picName in os.listdir(picDir):
        im = cv2.imread(picDir+picName)
        result = detector.detect(im)
        addRectangle(im,result)
        cv2.imshow('im',im)
        if cv2.waitKey(0)==27:
            break
    
if __name__=="__main__":
    run()
