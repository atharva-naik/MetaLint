 #!/usr/bin/env python
import os,time
import cv2
import numpy as np
import json
import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def load_result(result_,thr=0.5):
    result=[]
    global total_det
    for i in result_:
        boxes=[]
        i=i.strip().split('\t')
        face=i[0].split(" ")
        img=(face[0].split("/"))[1]
        for index in range(1,len(face),5):
            score=float(face[index+4])
            if score >= thr:
                total_det +=1
                boxes.append((float(face[index]),float(face[index+1]),float(face[index+2]),float(face[index+3])))
        result.append((img,boxes))
    return result

def load_gt(img):
    global gt_boxes_num
    label_file_path = os.path.join(label_path,(img.split("."))[0] + '.json')
    label_file=json.loads(open(label_file_path).read())
    persons = label_file['persons']
    #label框存入list
    label_list = []
    for item in persons:
        b_b = []
        b_b.append(item['body']['bbox'][0])
        b_b.append(item['body']['bbox'][1])
        b_b.append(item['body']['bbox'][0] + item['body']['bbox'][2])
        b_b.append(item['body']['bbox'][1] + item['body']['bbox'][3])
        label_list.append(b_b)
        gt_boxes_num += 1
    return label_list

def recal_and_precison(faces_boxes):
    global tp_num
    #计算指标
    for faces in faces_boxes:
        img=faces[0]
        #读取label
        label_list=load_gt(img)

        for face in faces[1]:
            # det_box=[face[0],face[1],face[2],face[3]]
            #算法结果框与gt框遍历
            iou_list=[]
            for index_, instance in enumerate(label_list):
                iou = bb_intersection_over_union(instance, face)
                iou_list.append(iou)
            max_iou_index = iou_list.index(max(iou_list))
            if iou_list[max_iou_index] >= 0.5:
                tp_num += 1
    #结果打印
    R = tp_num * 1.0 / gt_boxes_num
    P = tp_num * 1.0 / total_det
    return R,P,(P*R*2.0 / (P+R)),tp_num,total_det,gt_boxes_num

def cal_r_p():
    #各项数据保存
    result={}
    recall=[]
    precision=[]
    F1_score=[]
    threshould=[]
    tp_num_list=[]
    total_det_list=[]
    total_gt_list=[]
    for thr in range(50,100,10):
        global gt_boxes_num,tp_num,total_det
        gt_boxes_num=0   #gt框总数
        tp_num=0   #检测正确的数
        total_det=0 #检测框总数
        result_=open(result_path,'r+').readlines() #读取result文件  
        thr=thr/100
        #读取result
        faces_boxes=load_result(result_,thr)
        #计算指标
        r,p,f1,tp,det,gt=recal_and_precison(faces_boxes)
        threshould.append(thr)
        recall.append(r)   
        precision.append(p)
        F1_score.append(f1)
        tp_num_list.append(tp)
        total_det_list.append(det)
        total_gt_list.append(gt)
    result={
        "threshould":threshould,
        "recall":recall,
        "precision":precision,
        "fi_score":F1_score,
        "tp_num":tp_num_list,
        "total_det_num":total_det_list,
        "total_gt_num":total_gt_list
        }
    paint_percent(threshould,'threshould','percent','det_body_ssd',precision=precision,recall=recall)
    return result

#保存结果到本地
def save_result(data):
    df=pd.DataFrame.from_dict(data)     
    df.to_excel(save_path+save_name+".xlsx")  #(save_path+save_name+"_"+str(date)+".xlsx")

#画图
def paint_percent(thr_list, x_name='threshould', y_name='percent',pic_name='det_face', **args):
    plt.xlabel(x_name)
    plt.ylabel(y_name)     
    plt.title(pic_name)
    for k,v in args.items():
        plt.plot(thr_list, v, "x-", label=k)
    plt.grid(True)     
    plt.legend(bbox_to_anchor=(1.0,1), loc=1, borderaxespad=0.)
    plt.subplots_adjust(wspace = 0)
    plt.savefig(save_path+pic_name+".png",dpi=1000)

if __name__ == "__main__": 
    
    #外部传参
    parser = argparse.ArgumentParser()
    parser.add_argument("-result_path",default='result.txt')
    parser.add_argument("-label_path",default='test_data/px_json/') #标注文件所在路径
    parser.add_argument("-img_path",default='test_data/px_img/')    #图片所在路径，画框需要原始图片
    parser.add_argument("-save_path",default='result/') #结果保存路径
    #默认阈值，抠图
    args = parser.parse_args()
    global save_name,label_path,img_path,save_path
    result_path=args.result_path  #sdk结果文件路径
    save_name=((str(result_path).split("/")[-1]).split("."))[0]
    label_path=args.label_path
    img_path=args.img_path
    save_path=args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    #计算指标
    result =cal_r_p()

    #保存数据到excel,保存曲线
    save_result(result)
    
    #可添加画框功能，分类保存文件