import os
from mtcnn import detect_face
import caffe
import cv2

class IFaceZoneDetect:
    def __init__(self,model_dir,gpu_id):
        self.__minsize = 20
        self.__factor = 0.709

        self.__threshold = [0.6, 0.7, 0.7]


        self.__gpu_id = gpu_id
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)

        model_dir += '/' if not model_dir.endswith('/') else model_dir
        self.__pnet = caffe.Net(model_dir+"/mtcnn/det1.prototxt",model_dir+"/mtcnn/det1.caffemodel",caffe.TEST)
        self.__rnet = caffe.Net(model_dir+"/mtcnn/det2.prototxt",model_dir+"/mtcnn/det2.caffemodel",caffe.TEST)
        self.__onet = caffe.Net(model_dir+"/mtcnn/det3.prototxt",model_dir+"/mtcnn/det3.caffemodel",caffe.TEST)

    def detect(self,im):
        img_matlab = im.copy()
        tmp = img_matlab[:,:,2].copy()
        img_matlab[:,:,2] = img_matlab[:,:,0]
        img_matlab[:,:,0] = tmp
        boundingboxes, points = detect_face(img_matlab, self.__minsize, self.__pnet, self.__rnet, self.__onet, self.__threshold, False, self.__factor)

        return boundingboxes

def drawBoxes(im, boxes):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    for i in range(x1.shape[0]):
        cv2.rectangle(im, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0,255,0), 1)
    return im


if __name__=="__main__":
    model_dir = "/home/zqp/models"
    detector = IFaceZoneDetect(model_dir,0)


    pic_dir = "/home/zqp/project/mtcnn-master/"
    for picname in os.listdir(pic_dir):
        if not picname.endswith(".jpg"):
            continue

        im = cv2.imread(pic_dir+picname)

        box = detector.detect(im)
        drawBoxes(im,box)

        cv2.imshow("im",im)
        cv2.waitKey(0)






