# -*- coding:utf-8 -*- 
__author__ = 'xuy'

import cv2
import os

img=cv2.imread('RAP/RAP_dataset/CAM17_2014_03_18_20140318123432_20140318124019_tarid31_frame774_line2.png')

x_min=(511*168)//1280
y_min=(354*378)//720
x_max=(627*168)//1280
y_max=(528*378)//720



i1_pt1=(int(x_min),int(y_min))
i1_pt2=(int(x_max),int(y_max))

cv2.rectangle(img,pt1=i1_pt1,pt2=i1_pt2,thickness=3,color=(255,0,255))
cv2.imshow('Image',img)

cv2.waitKey(0)


