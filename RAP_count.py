# -*- coding:utf-8 -*- 
__author__ = 'xuy'
shirt_color={}
classes = ["Mixture", "Yellow", "Red", "Gray", "Orange", "White", "Purple", "Black", "Green", "Pink", "Blue", "Brown"]
with open('RAP/RAP_train.txt','r') as f:
    lines=f.readlines()
    for line in lines:
        line=line.strip()
        contents=line.split(' ')
        color_label=contents[1].split(',')[-1]
        shirt_color[color_label]=shirt_color.get(color_label,0)+1

for word in shirt_color:
    print('{} {}'.format(classes[int(word)],(shirt_color[word])))



'''
筛选之后的颜色类别
Black 10041
Yellow 603
White 1665
Purple 468
Pink 492
Orange 119
Red 1499
Mixture 2887
Brown 489
Green 707
Gray 2648
Blue 1722

'''
