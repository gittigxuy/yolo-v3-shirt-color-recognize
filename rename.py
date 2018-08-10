# -*- coding:utf-8 -*- 
__author__ = 'xuy'

import os
pic_dir=os.listdir("RAP/RAP_dataset/")
for temp in pic_dir:
    new_name=temp.replace('-','_')
    os.replace('RAP/RAP_dataset/'+temp,'RAP/RAP_dataset/'+new_name)
