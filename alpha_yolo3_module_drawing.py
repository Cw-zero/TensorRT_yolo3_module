# -*- coding: utf-8 -*-

import cv2
"""
alpha_yolo3模块的绘图函数

必须包含如下函数
drawing(frame,result_dict)

注意传递的字典应该遵循如下形式,
dict = {}
dict['info']={'frame_id':1,'camera_id':0}#info字段为图像的基本信息
dict['img']=@#@#@#@##@# #为opencv打开的numpy.arry
dict['data']={'number':1,'box_list':[[30,10,123,23]]} #data字段为算法产生的数据

"""


def drawing(frame,result_dict): 
	"""
	在输入的图像上按照result_dict中的信息进行绘图

	Parameters:
		frame: 图像 # opencv打开的图像,numpy.array格式
		result_dict: 字典,包含'info'字段与'data'字段,用来绘图

	Returns:
		None
	"""
	draw_bbx(frame,result_dict)

def draw_bbx(frame,result_dict):
	class_color_scheme = {'person':(0,255,255),'motorbike':(72,118,255),'car':(255,191,0),'bicycle':(0,128,255),'umbrella':(72,118,255),'truck':(152,251,152),'handbag':(255,165,0),'backpack':(160,32,240)}
	if 'box_list' in result_dict['data']:
		if 'class_list' in result_dict['data']:
			box_list = result_dict['data']['box_list']
			class_list = result_dict['data']['class_list']
			i = 0
			for j in range(len(box_list)):
				xmin = box_list[j][0]
				xmax = box_list[j][1]
				ymin = box_list[j][2]
				ymax = box_list[j][3]
				class_name = class_list[j]
				cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), class_color_scheme[class_name], 2)
				cv2.putText(frame, '{}_{}'.format(class_name,i), (xmin+10, ymin+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color_scheme[class_name])
				i+=1