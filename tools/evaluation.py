import os
import sys
sys.path.insert(0,os.getcwd())
import argparse

import copy
import random
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, \
    confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from numpy import mean
from tqdm import tqdm
from terminaltables import AsciiTable

import torch
# import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import time
import csv

from io import StringIO
from utils.dataloader import Mydataset, collate
from utils.train_utils import get_info, file2dict, set_random_seed
from models.build import BuildNet
from core.evaluations import evaluate
from utils.inference import init_model

# 评估结果、矩阵输出、类名、索引列表 、平均精度
def get_metrics_output(eval_results, metrics_output,classes_names, indexs, APs):
    f = open(metrics_output,'w+', newline='') #以追加的方式打开metrics_output文件
    writer = csv.writer(f) #创建一个csv写入器writer
    
    """
    输出并保存Accuracy、Precision、Recall、F1 Score、Confusion matrix结果
    """
    #创建一个列表p_r_f1
    p_r_f1 = [['Classes','Precision','Recall','F1 Score', 'Average Precision']]
    for i in range(len(classes_names)): #遍历每一个类
        data = []
        data.append(classes_names[i]) #将类名加入data列表
        data.append('{:.2f}'.format(eval_results.get('precision')[indexs[i]])) #从eval_results获取precision加入data列表
        data.append('{:.2f}'.format(eval_results.get('recall')[indexs[i]])) #从eval_results获取recall加入data列表
        data.append('{:.2f}'.format(eval_results.get('f1_score')[indexs[i]])) #从eval_results获取f1_score加入data列表
        data.append('{:.2f}'.format(APs[indexs[i]]*100)) #乘以100
        p_r_f1.append(data) #将data中的值加入到p_r_f1列表中
    #分类结果评估表格（第一张表）
    TITLE = 'Classes Results' #设置表格标题
    TABLE_DATA_1 = tuple(p_r_f1) #将p_r_f1列表转换为元组（tuple）
    table_instance = AsciiTable(TABLE_DATA_1,TITLE) #创建一个AsciiTable实例
    #table_instance.justify_columns[2] = 'right' 设置表格第三列右对齐
    print()
    print(table_instance.table) #打印Ascii表格
    writer.writerows(TABLE_DATA_1) #将表格（多行）写入csv文件
    writer.writerow([]) #写入一个空行
    print()

    #总体评估表格（第二张表）
    TITLE = 'Total Results' #设置表格标题
    TABLE_DATA_2 = (
    ('Top-1 Acc', 'Top-5 Acc', 'Mean Precision', 'Mean Recall', 'Mean F1 Score'),
    ('{:.2f}'.format(eval_results.get('accuracy_top-1',0.0)), '{:.2f}'.format(eval_results.get('accuracy_top-5',100.0)), '{:.2f}'.format(mean(eval_results.get('precision',0.0))),'{:.2f}'.format(mean(eval_results.get('recall',0.0))),'{:.2f}'.format(mean(eval_results.get('f1_score',0.0)))),
    )
    table_instance = AsciiTable(TABLE_DATA_2,TITLE) #创建一个AsciiTable实例
    #table_instance.justify_columns[2] = 'right' 设置表格第三列右对齐
    print(table_instance.table) #打印Ascii表格
    writer.writerows(TABLE_DATA_2) #将表格（多行）继续写入csv文件
    writer.writerow([]) #写入一个空行
    print()


    writer_list = []
    writer_list.append([' '] + [str(c) for c in classes_names]) #列表第一行：空格 类名 类名 类名 ...
    for i in range(len(eval_results.get('confusion'))): #遍历
        writer_list.append([classes_names[i]] + [str(x) for x in eval_results.get('confusion')[i]])
    #混淆矩阵（第三张表）
    TITLE = 'Confusion Matrix' #设置表格标题
    TABLE_DATA_3 = tuple(writer_list) #将writer_list列表转换为元组（tuple）
    table_instance = AsciiTable(TABLE_DATA_3,TITLE) #创建一个AsciiTable实例
    print(table_instance.table) #打印Ascii表格
    writer.writerows(TABLE_DATA_3) #将表格（多行）继续写入csv文件
    print()

    # #绘制混淆矩阵图
    # print(writer_list)
    # # matrix_path=f
    # confusion_matrix=pd.read_csv(f,skiprows=8,nrows=4)
    # print(confusion_matrix)
    # # confusion_matrix=pd.DataFrame(writer_list)
    # # print(confusion_matrix)
    # # confusion_matrix=np.array(writer_list)
    # # print(confusion_matrix)
    # plt.figure(figsize=(8,6))
    # sns.heatmap(confusion_matrix,annot=True,fmt='d',cmap='Blues')
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # plt.show()

# 模型预测结果、实际目标标签、图像路径列表、类别名称列表、类别索引列表、csv文件路径
def get_prediction_output(preds,targets,image_paths,classes_names,indexs,prediction_output):
    nums = len(preds) #计算预测结果的数量
    f = open(prediction_output,'a', newline='') #以追加的方式打开prediction_output文件
    writer = csv.writer(f) #创建一个csv写入器writer

    # 初始化列表results
    results = [['File', 'Pre_label', 'True_label', 'Success']]
    results[0].extend(classes_names) #第一行后面追加类名
    #遍历所有预测结果
    for i in range(nums):
        temp = [image_paths[i]] #初始化列表temp,并获取对应图像路径
        pred_label = classes_names[indexs[torch.argmax(preds[i]).item()]]
        true_label = classes_names[indexs[targets[i].item()]]
        success = True if pred_label == true_label else False
        class_score = preds[i].tolist() #把预测结果转换成列表形式
        temp.extend([pred_label,true_label,success]) #把数据写入temp列表
        temp.extend(class_score) #把数据写入temp列表
        results.append(temp) #把temp列表的数据加入results列表
        
    writer.writerows(results) #把results列表的多行数据写入csv文件

# 预测结果、实际目标标签、类别名称列表、保存结果的目录
def plot_ROC_curve(preds, targets, classes_names, savedir):
    rows = len(targets) #真实标签的数量
    cols = len(preds[0]) #预测结果中类别数量
    ROC_output = os.path.join(savedir, 'ROC')
    PR_output = os.path.join(savedir, 'P-R')
    os.makedirs(ROC_output)
    os.makedirs(PR_output) #创建两个文件夹目录ROC、P-R,保存曲线输出结果
    APs = [] #初始化列表APs,存储每个类别的平均精度
    for j in range(cols): #遍历预测结果中的每一个类
        gt, pre, pre_score = [], [], [] #初始化三个列表：真实标签、预测标签、预测分数
        for i in range(rows): #遍历真实标签
            if targets[i].item() == j: #如果真实标签=当前类别，则gt列表添加1，否则添加0
                gt.append(1)
            else:
                gt.append(0)
            
            if torch.argmax(preds[i]).item() == j: #如果预测结果中概率最高的类=当前类别，则pre列表添加1，否则添加0
                pre.append(1)
            else:
                pre.append(0)

            pre_score.append(preds[i][j].item()) #当前类别的预测分数加入pre_score列表

        # ROC
        ROC_csv_path = os.path.join(ROC_output,classes_names[j] + '.csv')
        ROC_img_path = os.path.join(ROC_output,classes_names[j] + '.png') #设置ROC曲线的路径
        ROC_f = open(ROC_csv_path,'a', newline='') #以追加的方式打开ROC_csv_path为文件
        ROC_writer = csv.writer(ROC_f) #csv写入器
        ROC_results = [] #创建一个ROC_results列表

        FPR,TPR,threshold=roc_curve(targets.tolist(), pre_score, pos_label=j) #通过roc_curve函数计算假正率、真正率、阈值

        AUC=auc(FPR,TPR) #auc函数计算ROC曲线下方的面积

        # 将AUC、FPR、TPR、Threshold写入csv文件
        ROC_results.append(['AUC', AUC])
        ROC_results.append(['FPR'] + FPR.tolist())
        ROC_results.append(['TPR'] + TPR.tolist())
        ROC_results.append(['Threshold'] + threshold.tolist())
        ROC_writer.writerows(ROC_results)

        #绘制ROC曲线
        plt.figure()
        plt.title(classes_names[j] + ' ROC CURVE (AUC={:.2f})'.format(AUC))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.01])
        plt.plot(FPR,TPR,color='g')
        plt.plot([0, 1], [0, 1], color='m', linestyle='--')
        plt.savefig(ROC_img_path) #保存图片

        # AP (gt为{0,1})
        AP = average_precision_score(gt, pre_score) #计算平均精度
        APs.append(AP)

        # P-R
        PR_csv_path = os.path.join(PR_output,classes_names[j] + '.csv')
        PR_img_path = os.path.join(PR_output,classes_names[j] + '.png')
        PR_f = open(PR_csv_path,'a', newline='')
        PR_writer = csv.writer(PR_f)
        PR_results = []
        
        PRECISION, RECALL, thresholds = precision_recall_curve(targets.tolist(), pre_score, pos_label=j)

        PR_results.append(['RECALL'] + RECALL.tolist())
        PR_results.append(['PRECISION'] + PRECISION.tolist())
        PR_results.append(['Threshold'] + thresholds.tolist())
        PR_writer.writerows(PR_results)

        plt.figure()
        plt.title(classes_names[j] + ' P-R CURVE (AP={:.2f})'.format(AP))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.01])
        plt.plot(RECALL,PRECISION,color='g')
        plt.savefig(PR_img_path)

    return APs

# def get_matrix_graph(eval_results, metrics_output,classes_names, indexs, APs):
#     #绘制混淆矩阵图
#     #print(writer_list)
#     # matrix_path=f
#     print(metrics_output)
#     confusion_matrix=pd.read_csv(metrics_output,skiprows=8,nrows=4)
#     print(confusion_matrix)
#     # confusion_matrix=pd.DataFrame(writer_list)
#     # print(confusion_matrix)
#     # confusion_matrix=np.array(writer_list)
#     # print(confusion_matrix)
#     plt.figure(figsize=(8,6))
#     sns.heatmap(confusion_matrix,annot=True,fmt='d',cmap='Blues')
#     plt.title('Confusion Matrix')
#     plt.xlabel('Predicted labels')
#     plt.ylabel('True labels')
#     plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--device', help='device used for training. (Deprecated)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    args = parser.parse_args()
    return args


def main(): 
    args = parse_args()
    model_cfg,train_pipeline,val_pipeline,data_cfg,lr_config,optimizer_cfg = file2dict(args.config)

    """
    创建评估文件夹、metrics文件、混淆矩阵文件
    """
    dirname = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) #生成一个目录名称，格式为：年-月-日-时-分-秒
    save_dir = os.path.join('eval_results',model_cfg.get('backbone').get('type'),dirname) #构建一条保存训练日志的目录路径：eval_results/'type'/dirname
    metrics_output = os.path.join(save_dir,'metrics_output.csv')
    prediction_output = os.path.join(save_dir,'prediction_results.csv') #生成两个用于评估的csv文件
    os.makedirs(save_dir)
    
    """
    获取类别名以及对应索引、获取标注文件
    """
    classes_map = 'datas/annotations.txt'  #获取类名和索引
    test_annotations  = 'datas/test.txt' #读取测试数据的标注文件
    classes_names, indexs = get_info(classes_map)
    with open(test_annotations, encoding='utf-8') as f: #用UTF-8编码打开测试文件
        test_datas   = f.readlines() #读取文件的所有行
    
    """
    设置各种随机种子确保结果可复现
    """
    set_random_seed(33, False)
    
    """
    生成模型、加载权重
    """
    # 选择评估使用的设备：指定的args.device/cuda/cpu
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 根据模型配置构建模型
    model = BuildNet(model_cfg)
    # 如果不是使用cpu，而是gpu进行训练，则用DataParallel包装模型以支持多gpu训练
    if device != torch.device('cpu'):
        model = DataParallel(model,device_ids=[args.gpu_id])

    model = init_model(model, data_cfg, device=device, mode='eval') #初始化模型
    
    """
    制作测试集并喂入Dataloader
    """
    val_pipeline = copy.deepcopy(val_pipeline)
    # 由于val_pipeline是用于推理，此处用做评估还需处理label
    val_pipeline = [data for data in val_pipeline if data['type'] != 'Collect']
    val_pipeline.extend([dict(type='ToTensor', keys=['gt_label']), dict(type='Collect', keys=['img', 'gt_label'])])
    
    test_dataset = Mydataset(test_datas, val_pipeline)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=data_cfg.get('batch_size'), num_workers=data_cfg.get('num_workers'), pin_memory=True, collate_fn=collate)
    
    """
    计算Precision、Recall、F1 Score、Confusion matrix
    """
    with torch.no_grad():
        preds,targets, image_paths = [],[],[]
        with tqdm(total=len(test_datas)//data_cfg.get('batch_size')) as pbar:
            for _, batch in enumerate(test_loader):
                images, target, image_path = batch
                outputs = model(images.to(device),return_loss=False)
                preds.append(outputs)
                targets.append(target.to(device))
                image_paths.extend(image_path)
                pbar.update(1)
                
    eval_results = evaluate(torch.cat(preds),torch.cat(targets),data_cfg.get('test').get('metrics'),data_cfg.get('test').get('metric_options'))
    
    APs = plot_ROC_curve(torch.cat(preds),torch.cat(targets), classes_names, save_dir) 
    get_metrics_output(eval_results,metrics_output,classes_names,indexs,APs)
    get_prediction_output(torch.cat(preds),torch.cat(targets),image_paths, classes_names, indexs, prediction_output)

    # # matrix_path=f
    # print(metrics_output)
    # confusion_matrix=pd.read_csv(metrics_output,skiprows=8,nrows=4)
    # print(confusion_matrix)
    # confusion_matrix.reset_index(drop=True)
    # print(confusion_matrix)
    #
    # # confusion_matrix=pd.DataFrame(writer_list)
    # # print(confusion_matrix)
    # # confusion_matrix=np.array(writer_list)
    # # print(confusion_matrix)
    # plt.figure(figsize=(8,6))
    # sns.heatmap(confusion_matrix,annot=True,fmt='d',cmap='Blues',xticklabels=confusion_matrix.columns,yticklabels=confusion_matrix.index)
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # plt.show()
           

if __name__ == "__main__":
    main()
