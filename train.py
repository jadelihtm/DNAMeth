# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
from logging import getLogger
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorboardX import SummaryWriter
from scipy.io import loadmat,savemat
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import sys
import torch.nn.functional as F
import cv2
from src.multicropdataset import MultiCropDataset_linear
import torchvision.transforms as transforms
from PIL import Image
from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    restart_from_checkpoint_domain,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
    accuracy,
)
import numpy as np
import copy
import src.resnet50 as resnet_models
from display.umap_display import dim_pca
from display.cal_cluster import eval_cluster_
logger = getLogger()
randomresized = transforms.Resize((166,166))
# human_labels=["Amy-Exc","ASC","CA1","CA3","CB","Chd7","DG","EC","Foxp2","HIP-Misc1","HIP-Misc2",\
#         "L23-IT","L4-IT","L56-NP","L5-ET","L5-IT","L6-CT","L6-IT","L6-IT-Car3","L6b",\
#             "Lamp5","Lamp5-Lhx6","MGC","MSN-D1","MSN-D2","ODC","OPC","PC","PKJ","Pvalb","Pvalb-ChC",\
#                 "Sncg","Sst","SubCtx-Cplx","THM-Exc","THM-MB","Vip","VLMC"]
human_labels=["ASC","L6-CT","EC","L23-IT","L4-IT","L5-IT","L6-IT","L6b","MGC","MSN-D2","L56-NP","ODC","OPC","PC","VLMC"]
lung_labels=["AT1","AT2","Basal","Blood cell","Ciliated","Endothelium","FABP4+ cell","Fibroblast",\
                    "Macrophage","Mucous_secretory","SMC"]
# mus_labels=["ASC","CGE-Lamp5","CGE-Vip","CLA","CT-L6","EC","IT-L23","IT-L4","IT-L5","IT-L6","L6b",\
#             "MGC","MGE-Pvalb","MGE-Sst","MSN-D2","NP-L6","ODC","OPC","PAL-Inh","PC","PT-L5",\
#                 "Unc5c","VLMC","VLMC-Pia"]
mus_labels=["ASC","CT-L6","EC","IT-L23","IT-L4","IT-L5","IT-L6","L6b",\
            "MGC","MSN-D2","NP-L6","ODC","OPC","PC","VLMC"]
target_datasets = "mus"
# 10+16,34+42,119+18,39+52,184+122,81+106,7+9,67+90,24+13,87+23,8,
weight = 1151./np.array([1151,1151,1151,1151,1151,1151,1151,1151,1151,1151,\
                         1151,1151,1151,1151,1151,1151,1151,1151,1151,1151,1151,\
                            1151,1151,1151,26,76,137,91,306,187,16,157,37,110,8]) # 8470
region=["Isocortex","Striatum","Olfactory"]
source_basedir = "/home/li_yu/rawdata/ATAC_Meth/human_snmcseq3/human_img_rmxy_5fold/"
target_basedir = "/home/li_yu/rawdata/ATAC_Meth/mus_brain/CroOrg_musbrain/"
# target_basedir = "/home/li_yu/rawdata/ATAC_Meth/human_lung/airway_tissue/CroOrg/"
# target_basedir = "/home/li_yu/rawdata/ATAC_Meth/human_snmcseq3/human_img_rmxy_5fold/Fold0"
# target_basedir = "/home/li_yu/rawdata/ATAC_Meth/mus_brain/mouse_atac/mos_allc_meth100kb_img_5fold/Fold0/"
raw_dump_path = "/home/li_yu/Proj01_Meth/swav_transfer_CroOrg/Trans2musoverlap_nat_human_domainnone_lr0.001ralpha1_test/"
parser = argparse.ArgumentParser(description="Evaluate models: Linear classification on ImageNet")

#########################
#### main parameters ####
#########################
parser.add_argument("--dump_path", type=str, default=raw_dump_path,
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")
# parser.add_argument("--data_path", type=str, default="/workspace/algorithm/lijia/liyu/DNAMeth/data/img100kMethSCAn",
#                     help="path to dataset repository")
# parser.add_argument("--test_data_path", type=str, default="/workspace/algorithm/lijia/liyu/DNAMeth/data/img100kMethSCAn_test",
#                     help="path to dataset repository")
# parser.add_argument("--pretrained", default="/home/li_yu/Proj01_Meth/swavDNA_linear/human100kbrmxy_linear_lr0.1_natpretrianed_5fold/", type=str, help="path to pretrained weights")
parser.add_argument("--pretrained", default="/home/li_yu/Proj01_Meth/swav_unsupervised/deepclusterv2_400ep_2x224_pretrain.pth.tar", type=str, help="path to pretrained weights")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")

#########################
#### model parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--global_pooling", default=True, type=bool_flag,
                    help="if True, we use the resnet50 global average pooling")
parser.add_argument("--use_bn", default=True, type=bool_flag,
                    help="optionally add a batchnorm layer before the linear classifier")

#########################
#### optim parameters ###
#########################                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--nesterov", default=False, type=bool_flag, help="nesterov momentum")
parser.add_argument("--scheduler_type", default="cosine", type=str, choices=["step", "cosine"])
# for multi-step learning rate decay
parser.add_argument("--decay_epochs", type=int, nargs="+", default=[30, 60],
                    help="Epochs at which to decay learning rate.")
parser.add_argument("--gamma", type=float, default=0.1, help="decay factor")
# for cosine learning rate schedule
parser.add_argument("--final_lr", type=float, default=0.001, help="final learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str,
                    help="url used to set up distributed training")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

def get_tick(cates,base=0,labelsname=None,label_tick=None):
    n = 0
    for icate in cates:
        labelsname[icate] = n+base
        label_tick.append(icate)
        n = n+1
    return labelsname,label_tick

def main(kfold,args):
    fix_random_seeds(args.seed)
    args.source_labelname = {}
    args.source_label_tick = []
    args.source_labelname,args.source_label_tick = get_tick(human_labels,0,args.source_labelname,args.source_label_tick)
    if target_datasets == "mus":
        # args.labelname = args.source_labelname
        # args.label_tick = args.source_label_tick
        args.labelname = {}
        args.label_tick = []
        args.labelname,args.label_tick = get_tick(mus_labels,0,args.labelname,args.label_tick)
        mean = [0.7890,0.7890,0.7890]
        std= [0.1981,0.1981,0.1981]
        target_batchsize = int(args.batch_size*10689/33898)
    else:
        # args.labelname = args.source_labelname
        # args.label_tick = args.source_label_tick
        args.labelname = {}
        args.label_tick = []
        args.labelname,args.label_tick = get_tick(lung_labels,0,args.labelname,args.label_tick)
        mean = [0.5064,0.5064,0.5064]
        std= [0.3142,0.3142,0.3142]
        target_batchsize = int(args.batch_size*19889/33898)
    logger, training_stats = initialize_exp(args, "epoch", "loss","cls_loss","domain_loss", "reg_loss")
    logdir = os.path.join(args.dump_path, 'logdir','train')
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir)
    testlogdir = os.path.join(args.dump_path, 'logdir','test')
    if not os.path.exists(testlogdir):
        os.makedirs(testlogdir)
    testwriter = SummaryWriter(log_dir=testlogdir)

    # build data
    train_source_dataset = MultiCropDataset_linear(
        args.source_data_path,
        args.size_crops,
        dataset="ratio",
        mean = [0.8555,0.8555,0.8555],
        std= [0.2396,0.2396,0.2396], ## human
        labelnames = args.source_labelname,
    )
    test_source_dataset = MultiCropDataset_linear(
        args.test_source_data_path,
        args.size_crops,
        dataset="ratio",
        return_index=True,
        mean = [0.8555,0.8555,0.8555],
        std= [0.2396,0.2396,0.2396], ## human
        labelnames = args.source_labelname
    )
    train_target_dataset = MultiCropDataset_linear(
        args.target_data_path,
        args.size_crops,
        dataset="ratio",
        mean = mean,
        std= std,
        labelnames = args.source_labelname,
        target_datasets=True,
    )
    test_target_dataset = MultiCropDataset_linear(
        args.test_target_data_path,
        args.size_crops,
        dataset="ratio",
        return_index=True,
        mean = mean,
        std= std,
        labelnames = args.source_labelname,
        target_datasets=True,
    )

    train_source_sampler = torch.utils.data.distributed.DistributedSampler(train_source_dataset)
    train_source_loader = torch.utils.data.DataLoader(
        train_source_dataset,
        sampler=train_source_sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False
    )
    test_source_sampler = torch.utils.data.distributed.DistributedSampler(test_source_dataset)
    test_source_loader = torch.utils.data.DataLoader(
        test_source_dataset,
        sampler=test_source_sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False
    )
    train_target_sampler = torch.utils.data.distributed.DistributedSampler(train_target_dataset)
    train_target_loader = torch.utils.data.DataLoader(
        train_target_dataset,
        sampler=train_target_sampler,
        batch_size=target_batchsize,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False
    )
    test_target_sampler = torch.utils.data.distributed.DistributedSampler(test_target_dataset)
    test_target_loader = torch.utils.data.DataLoader(
        test_target_dataset,
        sampler=test_target_sampler,
        batch_size=target_batchsize,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False
    )

    # build model
    model = resnet_models.__dict__[args.arch](output_dim=len(args.source_label_tick),cls=False, eval_mode=False)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # domain_discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(domain_discriminator)


    # model to gpu
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on],
        find_unused_parameters=True,
    )
    
    # set optimizer
    trunk_parameters = []
    head_parameters = []
    trunk_names = []
    head_names = []
    for name, param in model.named_parameters():
        if 'head' in name:
            head_parameters.append(param)
            head_names.append(name)
        else:
            trunk_parameters.append(param)
            trunk_names.append(name)
    optimizer = torch.optim.SGD(
        [{'params': trunk_parameters},
         {'params': head_parameters, 'lr': args.final_lr}],
        lr=args.lr,
        momentum=0.9,
        weight_decay=0,
    )
    # optimizer_dd = torch.optim.SGD(domain_discriminator.parameters(),lr=args.lr,momentum=0.9,weight_decay=0,)
    # set scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, args.decay_epochs, gamma=args.gamma
    )

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        # args.pretrained,
        # os.path.join(args.dump_path, "checkpoints","ckp-59.pth.tar"),
        # os.path.join(args.pretrained,"Fold%d/checkpoint.pth.tar"%kfold),
        # os.path.join(args.pretrained,"Fold"+str(kfold),"checkpoints","ckp-29.pth.tar"),
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        scheduler=scheduler,
        mode="test",
    )
    # for name,param in model.named_parameters():
    #     if not "prototypes" in name  and not "projection" in name:# and not "projection_head" in key: #(******)
    #         param.requires_grad = False
    # start_epoch = 0
    start_epoch = to_restore["epoch"]
    cudnn.benchmark = True
    if not os.path.exists(os.path.join(args.dump_path,'features')):
            os.mkdir(os.path.join(args.dump_path,'umap'))
            os.mkdir(os.path.join(args.dump_path,'features'))
            os.mkdir(os.path.join(args.dump_path,'umap','train'))
            os.mkdir(os.path.join(args.dump_path,'features','train'))
            os.mkdir(os.path.join(args.dump_path,'umap','test'))
            os.mkdir(os.path.join(args.dump_path,'features','test'))
            os.mkdir(os.path.join(args.dump_path,'umap','test_source'))
            os.mkdir(os.path.join(args.dump_path,'features','test_source'))
    
    # validate_network(test_source_loader,test_target_loader, model,domain_discriminator,ep=-1)


    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set samplers
        train_source_loader.sampler.set_epoch(epoch)
        train_target_loader.sampler.set_epoch(epoch)

        scores,acc_sr,acc_tr,label_train,embs,predlabel = train(model, optimizer, train_source_loader,train_target_loader,epoch)
        (features,embs_sr,embs_sr2,embs_tr2) = embs
        test_loss,test_acc,test_features,test_labelnames_sum,test_predlabel_sum\
                = validate_network(test_source_loader, test_target_loader, model,ep=epoch)
        training_stats.update(scores)
        (_,loss,cls_loss,domain_loss,reg_loss) = scores
        (labelnames,labelnames_sum) = label_train
        # ari_,nmi_,sil_ = eval_cluster(labelnames,predlabel,features)

        writer.add_scalar('loss', loss, (epoch+1)*len(train_source_loader))
        writer.add_scalar('reg_loss', reg_loss, (epoch+1)*len(train_source_loader))
        writer.add_scalar('cls_loss', cls_loss, (epoch+1)*len(train_source_loader))
        writer.add_scalar('domain_loss', domain_loss, (epoch+1)*len(train_source_loader))
        writer.add_scalar('lr', optimizer.param_groups[0]["lr"], (epoch+1)*len(train_source_loader))
        writer.add_scalar('lr_head', optimizer.param_groups[1]["lr"], (epoch+1)*len(train_source_loader))
        writer.add_scalar('acc_sr', acc_sr, (epoch+1)*len(train_source_loader))
        writer.add_scalar('acc_tr', acc_tr, (epoch+1)*len(train_source_loader))
        testwriter.add_scalar('acc_sr', test_acc, (epoch+1)*len(train_source_loader))
        testwriter.add_scalar('loss', test_loss, (epoch+1)*len(train_source_loader))
        # writer.add_scalar('sil', sil_, (epoch+1)*len(train_loader))

        scheduler.step()

        # save checkpoint
        print("Time to save checkpoint:")
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(save_dict, os.path.join(args.dump_path, "checkpoint.pth.tar"))
            if (epoch+1) % 20 == 0:
                torch.save({"epoch": epoch + 1,"state_dict":model.state_dict()}, os.path.join(args.dump_path, "checkpoints","ckp-%d.pth.tar"%epoch))
        if ((epoch+1) % 5 == 0):
            dim_pca(features,np.array(labelnames),os.path.join(args.dump_path,'umap','train'),epoch,args.label_tick)
            savemat(os.path.join(args.dump_path,'features','train','Ep%d_fea10kb2048-%s.mat'%(epoch,args.arch)),\
                    mdict={'epoch':epoch,'label_tr':np.array(labelnames),'label_sr':np.array(labelnames_sum),\
                           'fea_tr':features,'fea_sr':embs_sr,'fea_sr2':embs_sr2,'fea_tr2':embs_tr2,'predict':np.array(predlabel)})

            # test_loss,test_acc,test_features,test_labelnames_sum,test_predlabel_sum\
            #     = validate_network(test_source_loader, test_target_loader, model,ep=epoch)
    # validate_network(test_source_loader, test_target_loader, model,ep=100)
    idx = np.where(np.array(test_labelnames_sum)==20)[0]
    sil_test = eval_cluster_(np.array(test_predlabel_sum)[idx],np.array(test_features)[idx,:])
    
    logger.info(
            "Test:\t"
            "Acc@1 {top1:.3f}\t"
            "SIL {sil:.3f}\t".format(top1=test_acc,sil=sil_test))
    
    print("Done")



# Low entropy regularization for target domain samples (without labels)
def entropy_loss(predictions):
    """ predictions: predicted softmax output for target domain samples """ 
    # Compute softmax probabilities 
    softmax_probs = F.softmax(predictions, dim=1)
    # Compute entropy for each prediction 
    entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-6), dim=1) 
    # Return mean entropy 
    return torch.mean(entropy)

def train(model, optimizer, source_loader, target_loader, epoch):
    """
    Train the models on the dataset.
    """
    # running statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # training statistics
    losses = AverageMeter()
    top1 = AverageMeter()
    top1_source = AverageMeter()
    cls_losses  = AverageMeter()
    domain_losses = AverageMeter()
    reg_targets = AverageMeter()
    end = time.perf_counter()
    labelnames = []
    labelnames_sum = []
    embs= []
    embs_sr =[]
    embs_sr2 = []
    embs_tr2 = []
    predlabel = []

    model.train()
    criterion = nn.CrossEntropyLoss().cuda()
    # criterion1 = nn.CrossEntropyLoss(weight=torch.tensor(weight.astype(np.float32))).cuda()
    # criterion_domain = nn.BCEWithLogitsLoss().cuda()
    criterion_unknow = nn.NLLLoss()
    for iter_epoch,((source_data, source_labels), (target_data,target_labels)) in enumerate(zip(source_loader, target_loader)):
        # measure data loading time

        data_time.update(time.perf_counter() - end)

        # move to gpu
        source_data = source_data[0].cuda(non_blocking=True)
        source_labels = source_labels.cuda(non_blocking=True)
        target_data = target_data[0].cuda(non_blocking=True)
        target_labels = target_labels.cuda(non_blocking=True)



        # forward
        #with torch.no_grad():
        source_features, class_preds,source_domain_preds,source_features2,target_features, target_class_preds,target_domain_preds,target_features2\
              = model(source_data,target_data)
        # emb, output = reglog(output)

        # compute cross entropy loss
        cls_loss = criterion(class_preds, source_labels)
        domain_labels_source = torch.zeros(source_data.size(0)).long().cuda()  # 源域标签为0
        domain_labels_target = torch.ones(target_data.size(0)).long().cuda()  # 目标域标签为1

        domain_loss_source = criterion_unknow(source_domain_preds, domain_labels_source)
        domain_loss_target = criterion_unknow(target_domain_preds, domain_labels_target)
        # reg_unknow = criterion_unknow(target_domain_preds[:,1].reshape(-1),unknow_flag.to(torch.float32))+criterion_unknow(source_domain_preds[:,1].reshape(-1),source_unknow_flag.to(torch.float32))
        reg_target = entropy_loss(target_class_preds)
        domain_loss = domain_loss_source + domain_loss_target#*(source_data.size(0)/target_data.size(0))
        loss = cls_loss#+ domain_loss*0#+reg_target

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()
        # step
        optimizer.step()
        with torch.no_grad():
            acc1_source, _ = accuracy(class_preds, source_labels, topk=(1, 3))
            top1_source.update(acc1_source[0], source_data.size(0))
            acc1, _ = accuracy(target_class_preds, target_labels, topk=(1, 3))
            top1.update(acc1[0], target_data.size(0))
            _, pred = target_class_preds.topk(1, 1, True, True)
            pred = pred.reshape(-1)
            # print(pred)
            predlabel+= pred.detach().cpu().numpy().tolist()
            embs+=target_features.detach().cpu().numpy().tolist()
            embs_sr+=source_features.detach().cpu().numpy().tolist()
            embs_sr2+=source_features2.detach().cpu().numpy().tolist()
            embs_tr2+=target_features2.detach().cpu().numpy().tolist()
            labelnames+=get_targetlabel(target_labels.detach().cpu().numpy().tolist())
            labelnames_sum+=source_labels.detach().cpu().numpy().tolist()

        # update stats
        losses.update(loss.item(), source_data.size(0))
        cls_losses.update(cls_loss.item(), source_data.size(0))
        domain_losses.update(domain_loss.item(), source_data.size(0))
        reg_targets.update(reg_target.item(), target_data.size(0))


        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        # verbose
        if args.rank == 0 and iter_epoch % 50 == 0:
            logger.info(
                "Epoch[{0}] - Iter: [{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Cls Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t"
                "Reg unknown {reg_target.val:.4f} ({reg_target.avg:.4f})\t"
                "Domain Loss {domain_loss.val:.4f} ({domain_loss.avg:.4f})\t"
                "train target acc {top1.avg:.4f}\t"
                "train source acc {top1_source.avg:.4f}\t"
                "LR {lr}".format( 
                    epoch,
                    iter_epoch,
                    len(source_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    cls_loss=cls_losses,
                    reg_target=reg_targets,
                    domain_loss=domain_losses,
                    top1=top1,
                    top1_source=top1_source,
                    lr=optimizer.param_groups[1]["lr"],
                )
            )
    print("Pred - =========================================\n",np.array(predlabel).reshape(1,-1))

    return (epoch, losses.avg,cls_losses.avg,domain_losses.avg,reg_targets.avg),top1_source.avg,top1.avg,(labelnames,labelnames_sum),(embs,embs_sr,embs_sr2,embs_tr2),predlabel

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message: is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def get_targetlabel(target_labels):
    target_label = []
    # ref = {1:0,16:1,7:2,11:3,12:4,15:5,17:6,19:7,22:8,24:9,13:10,25:11,26:12,27:13,37:14}
    ref = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14}
    for i in target_labels:
        target_label.append(ref[i])
    return target_label

def validate_network(test_source_dataset, target_loader, model,ep):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top1_source = AverageMeter()
    losses=AverageMeter()
    embs = []
    embs2 = []
    labelnames_sum = []
    labelnames = []
    predlabel_sum = []
    predlabel = []
    embs_source = []
    embs_source2 = []

    # switch to evaluate mode
    model.eval()

    criterion = nn.CrossEntropyLoss().cuda()
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    log = Logger()
    log.open(os.path.join(args.dump_path,'test.txt'),mode='a')
    log2 = Logger()
    log2.open(os.path.join(args.dump_path,'test_source.txt'),mode='a')
    with torch.no_grad():
        end = time.perf_counter()
        for iter_epoch,((_,source_data, source_labels), (_,target_data,target_labels)) in enumerate(zip(test_source_dataset, target_loader)):
            # move to gpu
            source_data = source_data[0].cuda(non_blocking=True)
            source_labels = source_labels.cuda(non_blocking=True)
            target_data = target_data[0].cuda(non_blocking=True)
            target_labels = target_labels.cuda(non_blocking=True)

            # compute output
            # emb_target, output_target = model(target_data)
            source_features, class_preds,_,source_features2,emb_target, output_target,_,emb_target2 = model(source_data,target_data)
            cls_loss = criterion(class_preds, source_labels)
            with torch.no_grad():
                _, pred = class_preds.topk(1, 1, True, True)
                pred = pred.reshape(-1)
                predlabel+= pred.detach().cpu().numpy().tolist()
                _, pred_target = output_target.topk(1, 1, True, True)
                pred_target = pred_target.reshape(-1)
                predlabel_sum+=pred_target.detach().cpu().numpy().tolist()
                embs_source+=source_features.detach().cpu().numpy().tolist()
                embs_source2+=source_features2.detach().cpu().numpy().tolist()
                embs+=emb_target.detach().cpu().numpy().tolist()
                embs2+=emb_target2.detach().cpu().numpy().tolist()

            losses.update(cls_loss.item(), source_data.size(0))
            acc1, _ = accuracy(output_target, target_labels, topk=(1, 3))
            top1.update(acc1[0], target_data.size(0))
            acc1, _ = accuracy(class_preds, source_labels, topk=(1, 3))
            top1_source.update(acc1[0], source_data.size(0))

            # measure elapsed time
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()
            labelnames+=source_labels.detach().cpu().numpy().tolist() 
            labelnames_sum+=get_targetlabel(target_labels.detach().cpu().numpy().tolist())
    
    precision = precision_score(np.array(predlabel_sum), np.array(labelnames_sum), average="macro")
    recall = recall_score(np.array(predlabel_sum), np.array(labelnames_sum), average="macro")
    macro_f1 = f1_score(np.array(predlabel_sum), np.array(labelnames_sum), average="macro")
    log.write("Cell Num %d  Acc %.3f  Precision %.3f  Recall %.3f  Macro_f1 %.3f\n"%(i,top1.avg/100,precision,recall,macro_f1))
    dim_pca(embs,np.array(labelnames_sum),os.path.join(args.dump_path,'umap','test'),ep,args.label_tick)
    savemat(os.path.join(args.dump_path,'features','test','Ep%d_fea10kb2048-%s.mat'%(ep,args.arch)),\
            mdict={'fea_tr':embs,'fea_tr2':embs2,'label_tr':np.array(labelnames_sum),"predict":np.array(predlabel_sum)})

    precision = precision_score(np.array(predlabel), np.array(labelnames), average="macro")
    recall = recall_score(np.array(predlabel), np.array(labelnames), average="macro") 
    macro_f1 = f1_score(np.array(predlabel), np.array(labelnames), average="macro")
    log2.write("Cell Num %d  Acc %.3f  Precision %.3f  Recall %.3f  Macro_f1 %.3f\n"%(i,top1_source.avg/100,precision,recall,macro_f1))
    dim_pca(embs_source,np.array(labelnames),os.path.join(args.dump_path,'umap','test_source'),ep,args.source_label_tick)
    savemat(os.path.join(args.dump_path,'features','test_source','Ep%d_fea10kb2048-%s.mat'%(ep,args.arch)),\
            mdict={'fea_sr':embs_source,'fea_sr2':embs_source2,'label_sr':np.array(labelnames),"predict":np.array(predlabel)})

    return  losses.avg,top1.avg.item(),embs,labelnames_sum,predlabel_sum

if __name__ == "__main__":
    global args
    args = parser.parse_args()
    init_distributed_mode(args)
    if not os.path.exists(args.dump_path):
        os.mkdir(args.dump_path)
    for i in range(1):
        args.dump_path = raw_dump_path
        args.kfold = i
        args.dump_path = os.path.join(args.dump_path,"Fold"+str(args.kfold))
        if not os.path.exists(args.dump_path):
            os.mkdir(args.dump_path)
        print("================================================================= Fold%d ================================================================="%i)
        source_data_path = os.path.join(source_basedir,"Fold"+str(i),"train_overlap")
        target_data_path = os.path.join(target_basedir,"train_overlap")
        args.source_data_path = source_data_path
        args.target_data_path = target_data_path
        test_source_data_path = os.path.join(source_basedir,"Fold"+str(i),"test_overlap")
        test_target_data_path = os.path.join(target_basedir,"test_overlap")
        args.test_target_data_path = test_target_data_path
        args.test_source_data_path = test_source_data_path

        main(i,args)
