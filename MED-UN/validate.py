#coding:utf-8
# Hongliang Zhang, Dec 2019
#python face_detect.py --exp_id cu-net-4  --bs 1
from __future__ import division
from torch.autograd import Variable
import json
import pdb
import shutil
import scipy.io
import sys, os, time
from PIL import Image, ImageDraw
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn.parameter import Parameter

from options.train_options import TrainOptions
from data.face_bbx_ce import FACE
from data.picture_load import LoadPicture
from models.cu_net_csdn import create_cu_net 
from utils.util import AverageMeter
from utils.util import TrainHistoryFace, get_n_params, get_n_trainable_params, get_n_conv_params
from utils.visualizer import Visualizer
from utils.checkpoint import Checkpoint
from utils.logger import Logger
from utils.util import AdjustLR
from pylib import FaceAcc, Evaluation, HumanAug
cudnn.benchmark = True
from pylib import loss_val,dis_map
flip_index = np.array([[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9], # outline
            [17, 26], [18, 25], [19, 24], [20, 23], [21, 22], # eyebrow
            [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46], # eye
            [31, 35], [32, 34], # nose
            [48, 54], [49, 53], [50, 52], [59, 57], [58, 56], # outer mouth
            [60, 64], [61, 63], [67, 65]]) # inner mouth
def main():
    opt = TrainOptions().parse() 
    train_history = TrainHistoryFace()
    checkpoint = Checkpoint()
    visualizer = Visualizer(opt)
    exp_dir = os.path.join(opt.exp_dir, opt.exp_id)
    log_name = opt.vis_env + 'log.txt'
    visualizer.log_name = os.path.join(exp_dir, log_name)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    num_classes = 68
    layer_num = 4
    net = create_cu_net(neck_size=4, growth_rate=32, init_chan_num=128,class_num=num_classes, layer_num=layer_num,order=1, loss_num=layer_num)
    net = torch.nn.DataParallel(net).cuda()
    optimizer = torch.optim.RMSprop(net.parameters(), lr=opt.lr, alpha=0.99,
                                    eps=1e-8, momentum=0, weight_decay=0)
    img_folder = "/home/zb/zhanghl/dataset-300w"
    #加载验证的图片
    img=LoadPicture("/home/zb/zhanghl/Du-net-1/dataset/face.json", img_folder, is_train=False)
    #print(img.shape)
    
	
	
    #加载权重图
    map_val = dis_map.dis_map(64,64)
    #map_val = dis_map.dis_map(64,64,5)
    map_val=Variable(map_val)
    map_val=map_val.cuda()
  
  
    #加载模型
    checkpoints = torch.load("/home/zb/zhanghl/Du-net-1/cu-net4-csdn-loss-bs16-2.510-3-e2/lr-0.0005-39-model-best.pth.tar")  
    # two argv img_folder savepath
    net.load_state_dict(checkpoints['state_dict'])
    optimizer.load_state_dict(checkpoints['optimizer']) 
    net = net.cuda()
	# switch to evaluate mode
    net.eval()	
	#加载数据
    val_loader = torch.utils.data.DataLoader(
        FACE("/home/zb/zhanghl/Du-net-1/dataset/face.json","/home/zb/zhanghl/dataset-300w", is_train=False),
        batch_size=opt.bs, shuffle=False,
        num_workers=opt.nThreads, pin_memory=True)
    #print("train is long ..........")
    #print(len(train_loader))
    print("the valiatiion is long .........")
    print(type(val_loader))
    #print type(optimizer)  
    #得到预测值
    print("valation is staring ;; ;;;;;;;;;")
    #visualizer.log_path = os.path.join(opt.exp_dir, opt.exp_id, 'val_log.txt')
    val_loss, val_rmse, predictions = validate(val_loader, net,visualizer, num_classes, flip_index,map_val)    
    print("the val_loss is :{loss},the val_rms is :{rms}".format(loss=val_loss,rms=val_rmse))
    
	
    #Save prediction path
    save_pred_path = "/home/zb/zhanghl/Du-net-1/yuce/"
    #checkpoint.save_preds(predictions)
    #Save forecast
    preds = predictions.numpy()
    scipy.io.savemat(save_pred_path, mdict={'preds': preds})
    	
    #To display the effect, you need to set the number of displays. 
    #The visualization of the challenge set needs to correspond to the input data.
    for index,temp in enumerate(img):
        plt.figure("Image") # Image window name
        plt.imshow(temp)
        plt.axis('on') # Turn off the axis is off
        plt.title('image') # Image title
        plt.scatter(preds[index,:, 0], preds[index,:, 1], s=10, marker='.', c='r')
        #plt.pause(0.001)
        plt.show()
        if index>5:
	        break
    
	   
def validate(val_loader, net, visualizer, num_classes, flip_index, map_val):
    batch_time = AverageMeter()
    losses_det = AverageMeter()
    losses = AverageMeter()
    rmses0 = AverageMeter()
    rmses1 = AverageMeter()
    rmses2 = AverageMeter()
    inp_batch_list = []
    pts_batch_list = []
    predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2)

    end = time.time()
    for i, (inp, heatmap, pts, index, center, scale) in enumerate(val_loader):
        input_var = torch.autograd.Variable(inp, volatile=True)

        heatmap = heatmap.cuda(async=True)
        target_var = torch.autograd.Variable(heatmap)

        output1 = net(input_var)
        loss = 0
        for per_out in output1:
            tmp_loss = (per_out - target_var) ** 2
            loss = loss + tmp_loss.sum() / tmp_loss.numel()
        output = output1[-1].data.cpu()
        loss_val.get_loss(target_var,per_out,4,map_val)
        preds = Evaluation.final_preds(output, center, scale, [64, 64])
        rmse = np.sum(FaceAcc.per_image_rmse(preds.numpy(), pts.numpy())) / inp.size(0)
        rmses2.update(rmse, inp.size(0))
        """measure elapsed time"""
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item())
        loss_dict = OrderedDict( [('loss', losses.avg),
                                  ('rmse', rmses2.avg)] )
        for n in range(output.size(0)):
            predictions[index[n], :, :] = preds[n, :, :]
        
    return  losses.avg, rmses2.avg, predictions



if __name__ == '__main__':
    main()
