#coding=utf-8
import os
import caffe
import numpy as np
import cv2

def addRectangle(result,im):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for r in result:
        cv2.rectangle(im,(r[0],r[1]),(r[2],r[3]),(0,0,255),3)
        label = "1:"+str(r[4])
        cv2.putText(im,label,(r[0],r[1]),font,1.0,(0,255,0),2)

class IVehicleZoneDetect:
    def __init__(self,modelDir,gpu_id):
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)

        modelDir += '/' if not modelDir.endswith('/') else modelDir
        prototxt = modelDir+"vehicleZone/vehicleZoneD1deploy.prototxt"
        weightfile = modelDir+"vehicleZone/vehicleZoneD1model.dat"

        self.__net = caffe.Net(prototxt,weightfile,caffe.TEST)
        inputBlobShape =  self.__net.blobs['data'].data[0].shape

        self.__gpu_id = gpu_id
        self.__input_geometry = (inputBlobShape[-1],inputBlobShape[-2])
        self.__mean = np.array([[[103.939,116.979,123.68]]],dtype=np.float32)
        self.__confidence_threshold = 0.3

    def __getInputBlob(self,im):
        im_org = im.astype(np.float32,copy=True) - self.__mean
        im_resize = cv2.resize(im_org,self.__input_geometry)
        blob = np.zeros((1,self.__input_geometry[1],self.__input_geometry[0],3),dtype=np.float32)
        blob[0,:,:,:] = im_resize
        channel_swap = (0,3,1,2)
        blob = blob.transpose(channel_swap)
        blob = blob.astype(np.float32)

        return blob

#im: bgr, uint8
    def detect(self,im):
        result = []
        blob = self.__getInputBlob(im)
        self.__net.blobs['data'].data[...] = blob

        h,w = im.shape[0:2]

        blobs_out = self.__net.forward()
        blobs_out = blobs_out['detection_out'].reshape((-1,7))
        for blob in blobs_out:
            if blob[0]==-1 or blob[2]<self.__confidence_threshold:
                continue

            x1 = int(blob[3]*w) if blob[3]*w>0 else 0
            y1 = int(blob[4]*h) if blob[4]*h>0 else 0
            x2 = int(blob[5]*w) if blob[5]*w<w else w-1
            y2 = int(blob[6]*h) if blob[6]*h<h else h-1

            result.append([x1,y1,x2,y2,blob[2]])

        return result

def run():
    modelDir = r"/home/zqp/install_lib/vehicleDll/models"
    detector = IVehicleZoneDetect(modelDir,0)

    picDir = r'/media/zqp/新加卷/dataSet/data/carData5/data_3/'
    cv2.namedWindow("im",0)
    for picName in os.listdir(picDir):
        im = cv2.imread(picDir+picName)
        result = detector.detect(im)
        addRectangle(result,im)


        cv2.imshow('im',im)
        if cv2.waitKey(0)==27:
            break

if __name__=="__main__":
    run()
