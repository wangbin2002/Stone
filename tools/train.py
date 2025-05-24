import os
import sys

# from typing import Sequence
sys.path.insert(0, os.getcwd())  #将当前目录放到sys.path列表的最前面（index:0）,导入模块时优先考虑当前模块
import copy
import argparse
import shutil
import time
import numpy as np
import random

import torch
# import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel

from utils.history import History
from utils.dataloader import Mydataset, collate
from utils.train_utils import train, validation, print_info, file2dict, init_random_seed, set_random_seed, resume_model
from utils.inference import init_model
from core.optimizers import *
from models.build import BuildNet


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')  #ArgumentParser参数解析器
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--device', help='device used for training. (Deprecated)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed training)')
    parser.add_argument(
        '--split-validation',
        action='store_true', #如果用户提供了该参数，相应的值被设置为Ture
        help='whether to split validation set from training set.')
    parser.add_argument(
        '--ratio',  #验证集与训练集的比例
        type=float,
        default=0.2,
        help='the proportion of the validation set to the training set.')
    parser.add_argument(
        '--deterministic',  #是否为 CUDNN后端设置确定性选项
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()  #解析命令行参数
    if 'LOCAL_RANK' not in os.environ:  #检查LOCAL_RANK的值是否存在，不存在则将其设置为args.local_rank的值
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    # 读取配置文件获取关键字段
    args = parse_args()  #调用parse_args()函数解析命令行参数
    #模型配置、训练管道、验证管道、数据配置、学习率配置、优化器配置
    model_cfg, train_pipeline, val_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(args.config)  #调用file2dict()将配置文件转换为字典，并获取对应信息
    print_info(model_cfg) #调用print_info()输出模型配置信息

    # 初始化
    meta = dict()  #初始化字典meta存储元数据
    dirname = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) #生成一个目录名称，格式为：年-月-日-时-分-秒
    save_dir = os.path.join('logs', model_cfg.get('backbone').get('type'), dirname) #构建一条保存训练日志的目录路径：logs/'type'/dirname
    meta['save_dir'] = save_dir #将目录存储在meta字典中

    # 设置随机数种子
    seed = init_random_seed(args.seed) #初始化随机数生成器的种子
    set_random_seed(seed, deterministic=args.deterministic) #设置随机数种子
    meta['seed'] = seed #将种子存储在meta字典中

    # 读取训练&制作验证标签数据
    total_annotations = "datas/train.txt" #指定训练数据的注释文件路径train.txt
    with open(total_annotations, encoding='utf-8') as f: #读取训练数据文件
        total_datas = f.readlines() #将训练数据的内容存储在total_datas列表
    if args.split_validation: #检查是否需要从训练集中分割出验证集
        total_nums = len(total_datas) #计算训练数据的总数
        # indices = list(range(total_nums))
        if isinstance(seed, int): #判断种子是否为整数
            rng = np.random.default_rng(seed)
            rng.shuffle(total_datas) #用种子打乱数据顺序
        val_nums = int(total_nums * args.ratio) #验证集的大小
        folds = list(range(int(1.0 / args.ratio)))
        fold = random.choice(folds)
        val_start = val_nums * fold
        val_end = val_nums * (fold + 1) #交叉验证起始索引、结束索引
        train_datas = total_datas[:val_start] + total_datas[val_end:] #分割训练集和验证集
        val_datas = total_datas[val_start:val_end] #验证集取中间，训练集取两边
    else: #不分割验证集
        train_datas = total_datas.copy() #train.txt全都作为训练集
        test_annotations = 'datas/test.txt' #指定测试数据的注释文件路径test.txt
        with open(test_annotations, encoding='utf-8') as f: #读取测试集数据
            val_datas = f.readlines() #将测试集数据存储在val_datas列表中

    # 初始化模型,详见https://www.bilibili.com/video/BV12a411772h
    # 选择训练使用的设备：指定的args.device/cuda/cpu
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Initialize the weights.')
    model = BuildNet(model_cfg) #根据模型配置构建模型
    if not data_cfg.get('train').get('pretrained_flag'): #如果没有指定预训练权重，则初始化模型的权重
        model.init_weights()
    if data_cfg.get('train').get('freeze_flag') and data_cfg.get('train').get('freeze_layers'): #根据数据配置看是否执行冻结操作
        freeze_layers = ' '.join(list(data_cfg.get('train').get('freeze_layers')))
        print('Freeze layers : ' + freeze_layers)
        model.freeze_layers(data_cfg.get('train').get('freeze_layers'))

    # 如果不是使用cpu，而是gpu进行训练，则用DataParallel包装模型以支持多gpu训练
    if device != torch.device('cpu'):
        model = DataParallel(model, device_ids=[args.gpu_id])

    # 初始化优化器
    optimizer = eval('optim.' + optimizer_cfg.pop('type'))(params=model.parameters(), **optimizer_cfg)

    # 初始化学习率更新策略
    lr_update_func = eval(lr_config.pop('type'))(**lr_config)

    # 制作数据集->数据增强&预处理,详见https://www.bilibili.com/video/BV1zY4y167Ju
    train_dataset = Mydataset(train_datas, train_pipeline)
    val_pipeline = copy.deepcopy(train_pipeline)
    val_dataset = Mydataset(val_datas, val_pipeline)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=data_cfg.get('batch_size'),
                              num_workers=data_cfg.get('num_workers'), pin_memory=True, drop_last=True,
                              collate_fn=collate)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=data_cfg.get('batch_size'),
                            num_workers=data_cfg.get('num_workers'), pin_memory=True,
                            drop_last=True, collate_fn=collate)

    # 将关键字段存储，方便训练时同步调用&更新
    runner = dict(
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        iter=0,
        epoch=0,
        max_epochs=data_cfg.get('train').get('epoches'),
        max_iters=data_cfg.get('train').get('epoches') * len(train_loader),
        best_train_loss=float('INF'),
        best_val_acc=float(0),
        best_train_weight='',
        best_val_weight='',
        last_weight=''
    )
    meta['train_info'] = dict(train_loss=[],
                              val_loss=[],
                              train_acc=[],
                              val_acc=[])

    # 是否从中断处恢复训练
    if args.resume_from:
        model, runner, meta = resume_model(model, runner, args.resume_from, meta)
    else:
        os.makedirs(save_dir)
        shutil.copyfile(args.config, os.path.join(save_dir, os.path.split(args.config)[1]))
        model = init_model(model, data_cfg, device=device, mode='train')

    # 初始化保存训练信息类
    train_history = History(meta['save_dir'])

    # 记录初始学习率，详见https://www.bilibili.com/video/BV1WT4y1q7qN
    lr_update_func.before_run(runner)

    # 训练
    for epoch in range(runner.get('epoch'), runner.get('max_epochs')):
        lr_update_func.before_train_epoch(runner)
        train(model, runner, lr_update_func, device, epoch, data_cfg.get('train').get('epoches'), data_cfg.get('test'),
              meta)
        validation(model, runner, data_cfg.get('test'), device, epoch, data_cfg.get('train').get('epoches'), meta)

        train_history.after_epoch(meta)  #画图


if __name__ == "__main__":
    main()
