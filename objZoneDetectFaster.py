#coding=utf-8
import caffe
import cv2
import numpy as np
import os
from bbox_transform import *
from nms import *

class ObjZoneDetectFaster:
    def __init__(self,prototxt,weightfile,gpu_id,targetSize,maxSize,confidence_threshold):
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        self.__net = caffe.Net(prototxt,weightfile,caffe.TEST)
        self.__pixel_mean = np.array([[[102.9801, 115.9465, 122.7717]]],dtype=np.float32)
        self.__targetSize = targetSize
        self.__maxSize = maxSize
        self.__confidence_threshold = confidence_threshold
        self.__class_num = self.__net.blobs['cls_prob'].data.shape[-1]

    def __getResizeScale(self,im):
        im_size_min = np.min(im.shape[0:2])
        im_size_max = np.max(im.shape[0:2])
        im_scale = float(self.__targetSize)/float(im_size_min)
        if np.round(im_scale*im_size_max)>self.__maxSize:
            im_scale = float(self.__maxSize)/float(im_size_max)

        return im_scale

    def __getInputBlob(self,im):
        im_scale = self.__getResizeScale(im)
        im_orig = im.astype(np.float32,copy=True) - self.__pixel_mean
        im_resize = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,interpolation=cv2.INTER_LINEAR)

        blob = np.zeros((1,im_resize.shape[0],im_resize.shape[1],3),dtype=np.float32)
        blob[0,:,:,:] = im_resize
        channel_swap = (0, 3, 1, 2)
        blob = blob.transpose(channel_swap)
        im_info = np.array([[blob.shape[2],blob.shape[3],im_scale]],dtype=np.float32)
        blob = blob.astype(np.float32)

        return blob,im_info

    def detect(self,im):
        blob,im_info = self.__getInputBlob(im)
        self.__net.blobs['data'].reshape(*(blob.shape))
        self.__net.blobs['im_info'].reshape(*(im_info.shape))
        forward_kwargs = {'data': blob.astype(np.float32, copy=False)}
        forward_kwargs['im_info'] = im_info.astype(np.float32, copy=False)

        blobs_out = self.__net.forward(**forward_kwargs)

        rois = self.__net.blobs['rois'].data.copy()
        boxes = rois[:, 1:5] / im_info[0][2]
        boxes = boxes.reshape((boxes.shape[:2]))
        scores = blobs_out['cls_prob']
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)

        pred_boxes = clip_boxes(pred_boxes, im.shape)

        result = self.__getDetectResult(scores,pred_boxes)

        return result

    def __getDetectResult(self,scores,boxes):
        result = {}
        NMS_THRESH = 0.3
        for cls_ind in range(self.__class_num-1):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)

            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]

            inds = np.where(dets[:, -1] >= self.__confidence_threshold)[0]
            if len(inds) == 0:
                continue

            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]

                if not result.has_key(cls_ind):
                    result[cls_ind] = []

                result[cls_ind].append([bbox[0],bbox[1],bbox[2],bbox[3],score])

        return result

def run():
    prototxt = r'/home/zqp/install_lib/vehicleDll/models/vehicleStructure/vehicleStructD3deploy.prototxt'
    weightfile = r'/home/zqp/install_lib/vehicleDll/models/vehicleStructure/vehicleStructD3model.dat'

    detector = ObjZoneDetectFaster(prototxt,weightfile,0,600,1000,0.7)
    picDir = r'/media/zqp/新加卷/dataSet/data/vehicleHead/carHeadType/奥迪_一汽大众奥迪_奥迪100or红旗小红旗_200x款/'
    for picName in os.listdir(picDir):
        im = cv2.imread(picDir+picName)
        detector.detect(im)
        cv2.imshow('im',im)
        if cv2.waitKey(0)==27:
            break

if __name__ == '__main__':
    run()



