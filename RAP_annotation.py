# -*- coding:utf-8 -*- 
__author__ = 'xuy'
'''
对于xml文件进行解析，所需要的类别是：颜色，以及人的位置
转换成RAP.txt文件

格式：
路径 xmin xmax ymin ymax label
这里label 就是颜色
'''

import xml.etree.ElementTree as ET
from os import getcwd
import os
import shutil

ImgPath='RAP/RAP_dataset/'
classes = ["Mixture", "Yellow", "Red", "Gray", "Orange", "White", "Purple", "Black", "Green", "Pink", "Blue", "Brown"]
multi_count=0
def convert_annotation(image_id,list_file):
    in_file=open('annotations/%s.xml'%image_id)
    tree = ET.parse(in_file)
    root = tree.getroot()

    count=0

    for obj in root.iter('object'):
        cls=obj.find('name').text
        if cls[:2]=='up':
            color_cls=cls[3:]
            if color_cls not in classes:
                continue
            color_cls_id = classes.index(color_cls)
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
                 int(xmlbox.find('ymax').text))
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(color_cls_id))
            count=count+1



imagelist=os.listdir(ImgPath)
list_file=open('RAP/RAP_train.txt','w')
for image in imagelist:
    image_pre,ext=os.path.splitext(image)

    list_file.write('RAP/RAP_dataset/%s.png'%(image_pre))
    convert_annotation(image_pre,list_file)
    list_file.write('\n')

list_file.close()
# print(multi_count)#18107

#删除掉多个颜色标签，仅仅保留一种颜色标签
with open('RAP/RAP_train.txt','r') as old_file:
    with open('RAP/RAP_train_new.txt','w')as new_file:
        lines = old_file.readlines()
        for line in lines:
            line=line.strip()
            contents=line.split()
            if len(contents)==2:
                multi_count+=1
                new_file.write(line)
                new_file.write('\n')

shutil.move('RAP/RAP_train_new.txt','RAP/RAP_train.txt')

print(multi_count)


















