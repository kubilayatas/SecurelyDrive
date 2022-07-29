import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadWebcam
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

####################   Driver Gaze Classifier   #########################

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import copy
from collections import namedtuple
import os
import random
import shutil
import time
import torch

class ResNet(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()
                
        block, n_blocks, channels = config
        self.in_channels = channels[0]
            
        assert len(n_blocks) == len(channels) == 4
        
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride = 2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride = 2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, output_dim)
        
    def get_resnet_layer(self, block, n_blocks, channels, stride = 1):
    
        layers = []
        
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False
        
        layers.append(block(self.in_channels, channels, stride, downsample))
        
        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels
            
        return nn.Sequential(*layers)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)
        
        return x, h


class BasicBlock(nn.Module):
    
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()
                
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, 
                               stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, 
                               stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace = True)
        
        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, 
                             stride = stride, bias = False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None
        
        self.downsample = downsample
        
    def forward(self, x):
        
        i = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample is not None:
            i = self.downsample(i)
                        
        x += i
        x = self.relu(x)
        
        return x

class Bottleneck(nn.Module):
    
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()
    
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, 
                               stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, 
                               stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size = 1,
                               stride = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)
        
        self.relu = nn.ReLU(inplace = True)
        
        if downsample:
            conv = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size = 1, 
                             stride = stride, bias = False)
            bn = nn.BatchNorm2d(self.expansion * out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None
            
        self.downsample = downsample
        
    def forward(self, x):
        
        i = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
                
        if self.downsample is not None:
            i = self.downsample(i)
            
        x += i
        x = self.relu(x)
    
        return x
#############################################################

class LSTM(nn.Module):
    
    def __init__(self,input_dim,hidden_dim,output_dim,layer_num):
        super(LSTM,self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = torch.nn.LSTM(input_dim,hidden_dim,layer_num,batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim,output_dim)
        self.bn = nn.BatchNorm1d(1) # 32 idi 1 yaptım
        
    def forward(self,inputs):
        x = self.bn(inputs)
        lstm_out,(hn,cn) = self.lstm(x)
        out = self.fc(lstm_out[:,-1,:])
        return out

#############################################################
def elestiem(pred,names,im0shape,imgshape):
    for i, det in enumerate(pred):
        if len(det)==0:
            df = pd.DataFrame(columns=['Class','Xc','Yc','W','H'])
        else:
            gn = torch.tensor(im0shape)[[1, 0, 1, 0]]
            det[:, :4] = scale_coords(imgshape[2:], det[:, :4], im0shape).round()
            lst=[]
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                line = (int(cls.tolist()), *xywh, conf) if opt.save_conf else (int(cls.tolist()), *xywh)
                cl,xc,yc,ww,hh = list(line)
                lst.append([cl,xc,yc,ww,hh])
            df = pd.DataFrame(lst,columns=['Class','Xc','Yc','W','H'])
        
    lstmdat = [] #if names[int(cls)] == "DriverFace":
    
    for i in range(0,len(names)):
        if names[i] == "Cigarette":
            fd = df[df['Class']==i]
            fd['Prox'] = abs(fd.Xc.astype(float)-0.0)/abs(fd.Xc.astype(float)-1.0)
            fd = fd.sort_values(by=['Prox'],axis=0)
            fd = fd[fd['Prox']<0.5]
            fd.reset_index(drop=True,inplace=True)
            if len(fd)==0:
                lstmdat.append(0.0) # x coord
                lstmdat.append(1.0) # y coord
            else:
                lstmdat.append(float(fd['Xc'][0])) # Right Hand x coord
                lstmdat.append(float(fd['Yc'][0])) # Right Hand y coord
        elif names[i] == "Cellphone":
            fd = df[df['Class']==i]
            fd['Prox'] = abs(fd.Xc.astype(float)-0.0)/abs(fd.Xc.astype(float)-1.0)
            fd = fd.sort_values(by=['Prox'],axis=0)
            fd = fd[fd['Prox']<0.6]
            fd.reset_index(drop=True,inplace=True)
            if len(fd)==0:
                lstmdat.append(0.0) # x coord
                lstmdat.append(1.0) # y coord
            else:
                lstmdat.append(float(fd['Xc'][0])) # Right Hand x coord
                lstmdat.append(float(fd['Yc'][0])) # Right Hand y coord
            
        elif names[i] == "DriverHand":
            fd = df[df['Class']==i]
            fd.reset_index(drop=True,inplace=True)
            if len(fd)==0:
                lstmdat.append(0.0) # Left  Hand x coord
                lstmdat.append(1.0) # Left  Hand y coord
                lstmdat.append(0.0) # Right Hand x coord
                lstmdat.append(1.0) # Right Hand y coord
            elif len(fd)==1:
                lstmdat.append(0.0) # Left  Hand x coord
                lstmdat.append(1.0) # Left  Hand y coord
                lstmdat.append(float(fd['Xc'][0])) # Right Hand x coord
                lstmdat.append(float(fd['Yc'][0])) # Right Hand y coord
            else:
                fd['Prox'] = abs(fd.Xc.astype(float)-0.333)/abs(fd.Xc.astype(float)-0.0)
                fd = fd.sort_values(by=['Prox'],axis=0)
                lstmdat.append(float(fd['Xc'][1])) # Left  Hand x coord
                lstmdat.append(float(fd['Yc'][1])) # Left  Hand y coord
                lstmdat.append(float(fd['Xc'][0])) # Right Hand x coord
                lstmdat.append(float(fd['Yc'][0])) # Right Hand y coord
        else:
            fd = df[df['Class']==i]
            fd.reset_index(drop=True,inplace=True)
            if len(fd)==0:
                lstmdat.append(0.0) # 
                lstmdat.append(0.0) #
            else:
                lstmdat.append(float(fd['Xc'][0])) # 
                lstmdat.append(float(fd['Yc'][0])) #
    #lstmdat.append(1.0 if looking=="forward" else 0.0)
    return lstmdat
    #with open(lstm_txt_path + '.txt', 'a') as f:
    #    #f.write(('%g,' * len(lstmdat)).rstrip() % lstmdat + '\n')
    #    f.write(','.join([str(x) for x in lstmdat]) + '\n')
#############################################################

def detect(save_img=True):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.DetectorWeights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'nvarguscamerasrc'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / 'lstm_data' if opt.extract_lstmdata else save_dir).mkdir(parents=True, exist_ok=True)  # make lstm data dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    deviceLSTM = select_device(opt.deviceLSTM)
    deviceGAZE = select_device(opt.deviceGAZE)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    
    
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        if source.lower().startswith(('nvarguscamerasrc')):
            dataset = LoadWebcam(source, img_size=imgsz)
        else:
            dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        
        #lstm_dat_flag
        # Process detections
        LstmDetFlag = True
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = Path(path), '', im0s
            
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('_{:04d}'.format(dataset.frame-1) if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            _gaze = ""
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string 0: 320x416 1 Cellphones,

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if opt.drv_gaze:
                        if names[int(cls)] == "DriverFace":
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            xyxy_list = torch.tensor(xyxy).tolist()
                            x1 = xywh[0]*img.shape[3]-(xywh[2]*img.shape[3])/2
                            x2 = xywh[0]*img.shape[3]+(xywh[2]*img.shape[3])/2
                            y1 = xywh[1]*img.shape[2]-(xywh[3]*img.shape[2])/2
                            y2 = xywh[1]*img.shape[2]+(xywh[3]*img.shape[2])/2
                            trans = transforms.ToPILImage()
                            cropped_im = transforms.functional.crop(img,int(y1),int(x1),int(y2-y1),int(x2-x1))
                            cropped_im = cropped_im.to(deviceGAZE)
                            cropped_im = cropped_im[0,:,:,:]
                            cropped_im = trans(cropped_im)
                            #cropped_im.show()
                            looking = predict_DriverGaze(cropped_im)
                            _gaze = looking
                            im0 = cv2.putText(im0, "Looking: " + looking, (450, 20), 0, 0.5, [225, 0, 0], thickness=1, lineType=cv2.LINE_AA)
                            #print(looking)
                            
                            
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    
                    


            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            
            if opt.lstm_detect and LstmDetFlag:
                LstmDetFlag = False
                lstm_dat = elestiem(pred,names,im0.shape,img.shape)
                #_gaze = _gaze if '_gaze' in vars() or '_gaze' in globals() else ""
                lstm_dat.append(1.0 if _gaze=="forward" else 0.0)
                lstm_dat = torch.as_tensor([[lstm_dat]])
                lstm_dat.to(deviceLSTM)
                output = lstm_model(lstm_dat)
                #top_n, top_i = output.topk(1)
                output = torch.tensor([min(1.0,max(0.0,elem.item())) for elem in output[0]]).to(device)
                o = [(1 if elem.item()>=0.5 else 0) for elem in output]
                #category_i = top_i[0].item()
                lab = ''.join([str(elem) for elem in o])
                #print(lab)
                PhoneCall = bool(int(o[0]))
                Smoking = bool(int(o[1]))
                Texting = bool(int(o[2]))
                #print("burayı sonra sil")
                im0 = cv2.putText(im0, "Phone Call: " + ("True" if PhoneCall else "False"), (0, 20), 0, 0.5, ([0, 0, 255] if PhoneCall else [255, 0, 0]), thickness=1, lineType=cv2.LINE_AA)
                im0 = cv2.putText(im0, "   Smoking: " + ("True" if Smoking else "False"), (150, 20), 0, 0.5, ([0, 0, 225] if Smoking else [225, 0, 0]), thickness=1, lineType=cv2.LINE_AA)
                im0 = cv2.putText(im0, "   Texting: " + ("True" if Texting else "False"), (300, 20), 0, 0.5, ([0, 0, 225] if Texting else [225, 0, 0]), thickness=1, lineType=cv2.LINE_AA)
                im0 = cv2.putText(im0, "       Lab: " + lab, (0, 40), 0, 0.5, ([0, 0, 225] if Texting else [225, 0, 0]), thickness=1, lineType=cv2.LINE_AA)

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    if save_path.split('.')[-1]=='jpg':
                        cv2.imwrite(save_path, im0)
                    else:
                        cv2.imwrite(save_path + ".jpg", im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

            if opt.extract_lstmdata:# and len(det):
                lstm_txt_path = str(save_dir / 'lstm_data' / p.stem)
                lstmdat = [] #if names[int(cls)] == "DriverFace":
                lstmdat.append(p.stem + ("_{:04d}.jpg".format(dataset.frame-1) if dataset.mode == 'video' else ''))
                if len(det)==0:
                    df = pd.DataFrame(columns=['Class','Xc','Yc','W','H'])
                else:
                    with open(txt_path + '.txt', 'r') as f:
                        ddt = f.read()
                        ddt = ddt.split('\n')
                        ddt = list(filter(None, ddt))
                    lst = []
                    for i,dd in enumerate(ddt):
                        cl,xc,yc,ww,hh = dd.split(' ')
                        #print(nm,cl,xc,yc,ww,hh)
                        lst.append([cl,xc,yc,ww,hh])
                    df = pd.DataFrame(lst,columns=['Class','Xc','Yc','W','H'])
                for i in range(0,len(names)):
                    if names[i] == "Cigarette":
                        fd = df[df['Class']=='{}'.format(i)]
                        fd['Prox'] = abs(fd.Xc.astype(float)-0.0)/abs(fd.Xc.astype(float)-1.0)
                        fd = fd.sort_values(by=['Prox'],axis=0)
                        fd = fd[fd['Prox']<0.5]
                        fd.reset_index(drop=True,inplace=True)
                        if len(fd)==0:
                            lstmdat.append(0.0) # x coord
                            lstmdat.append(1.0) # y coord
                        else:
                            lstmdat.append(float(fd['Xc'][0])) # Right Hand x coord
                            lstmdat.append(float(fd['Yc'][0])) # Right Hand y coord
                    elif names[i] == "Cellphone":
                        fd = df[df['Class']=='{}'.format(i)]
                        fd['Prox'] = abs(fd.Xc.astype(float)-0.0)/abs(fd.Xc.astype(float)-1.0)
                        fd = fd.sort_values(by=['Prox'],axis=0)
                        fd = fd[fd['Prox']<0.6]
                        fd.reset_index(drop=True,inplace=True)
                        if len(fd)==0:
                            lstmdat.append(0.0) # x coord
                            lstmdat.append(1.0) # y coord
                        else:
                            lstmdat.append(float(fd['Xc'][0])) # Right Hand x coord
                            lstmdat.append(float(fd['Yc'][0])) # Right Hand y coord
                        
                    elif names[i] == "DriverHand":
                        fd = df[df['Class']=='{}'.format(i)]
                        fd.reset_index(drop=True,inplace=True)
                        if len(fd)==0:
                            lstmdat.append(0.0) # Left  Hand x coord
                            lstmdat.append(1.0) # Left  Hand y coord
                            lstmdat.append(0.0) # Right Hand x coord
                            lstmdat.append(1.0) # Right Hand y coord
                        elif len(fd)==1:
                            lstmdat.append(0.0) # Left  Hand x coord
                            lstmdat.append(1.0) # Left  Hand y coord
                            lstmdat.append(float(fd['Xc'][0])) # Right Hand x coord
                            lstmdat.append(float(fd['Yc'][0])) # Right Hand y coord
                        else:
                            fd['Prox'] = abs(fd.Xc.astype(float)-0.333)/abs(fd.Xc.astype(float)-0.0)
                            fd = fd.sort_values(by=['Prox'],axis=0)
                            lstmdat.append(float(fd['Xc'][1])) # Left  Hand x coord
                            lstmdat.append(float(fd['Yc'][1])) # Left  Hand y coord
                            lstmdat.append(float(fd['Xc'][0])) # Right Hand x coord
                            lstmdat.append(float(fd['Yc'][0])) # Right Hand y coord
                    else:
                        fd = df[df['Class']=='{}'.format(i)]
                        fd.reset_index(drop=True,inplace=True)
                        if len(fd)==0:
                            lstmdat.append(0.0) # 
                            lstmdat.append(0.0) #
                        else:
                            lstmdat.append(float(fd['Xc'][0])) # 
                            lstmdat.append(float(fd['Yc'][0])) #
                lstmdat.append(1.0 if looking=="forward" else 0.0)
                with open(lstm_txt_path + '.txt', 'a') as f:
                    #f.write(('%g,' * len(lstmdat)).rstrip() % lstmdat + '\n')
                    f.write(','.join([str(x) for x in lstmdat]) + '\n')

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print('Done. (%.3fs)' % (time.time() - t0))

def predict_DriverGaze(img,ImgSize=224):
    classes = ['forward', 'other']
    deviceGAZE = select_device(opt.deviceGAZE)
    means = torch.from_numpy(np.array([0.5743, 0.5249, 0.5142]))
    stds = torch.from_numpy(np.array([0.2522, 0.2462, 0.2368]))
    eval_transforms = transforms.Compose([
        transforms.Resize(ImgSize),
        transforms.ToTensor(),
        transforms.Normalize(mean = means,std = stds)
        ])
    img_normalized = eval_transforms(img).float()
    img_normalized = img_normalized.unsqueeze_(0)
    img_normalized = img_normalized.to(deviceGAZE)
    with torch.no_grad():
        y_pred, _ = DriverGazeModel(img_normalized)
        y_prob = F.softmax(y_pred, dim = -1)
        top_pred = y_prob.argmax(1, keepdim = True)
        prob = top_pred.cpu()
    return classes[prob]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DetectorWeights', nargs='+', type=str, default='ObjectDetectorModel.pt', help='model.pt path(s)')
    parser.add_argument('--DriverGazeWeights', nargs='+', type=str, default='DriverGazeModel.pt', help='model.pt path(s)')
    parser.add_argument('--LSTMWeights', nargs='+', type=str, default='LSTMmodel.pkl', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--deviceGAZE', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--deviceLSTM', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--extract-lstmdata', action='store_true', help='extract lstm data')    
    parser.add_argument('--lstm-detect', action='store_true', help='extract lstm data')    
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--drv-gaze', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    if opt.drv_gaze:
        ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
        resnet18_config = ResNetConfig(block = BasicBlock,n_blocks = [2,2,2,2],channels = [64, 128, 256, 512])
        resnet34_config = ResNetConfig(block = BasicBlock,n_blocks = [3,4,6,3],channels = [64, 128, 256, 512])
        resnet50_config = ResNetConfig(block = Bottleneck,n_blocks = [3, 4, 6, 3],channels = [64, 128, 256, 512])
        resnet101_config = ResNetConfig(block = Bottleneck,n_blocks = [3, 4, 23, 3],channels = [64, 128, 256, 512])
        resnet152_config = ResNetConfig(block = Bottleneck,n_blocks = [3, 8, 36, 3],channels = [64, 128, 256, 512])
        OUTPUT_DIM = 2
        DriverGazeModel = ResNet(resnet50_config, OUTPUT_DIM)
        DriverGazeModel.load_state_dict(torch.load(opt.DriverGazeWeights))
        DriverGazeModel.to(select_device(opt.deviceGAZE))
        DriverGazeModel.eval()
    
    if opt.lstm_detect:
        n_hidden = 128
        n_joints = 19
        #n_categories = 6
        regression_out = 3
        n_layer = 3
        lstm_model = LSTM(n_joints,n_hidden,regression_out,n_layer)
        lstm_model.load_state_dict(torch.load(opt.LSTMWeights[0]))
        lstm_model.to(select_device(opt.deviceLSTM))
        lstm_model.eval()
    
    LABELS = [
    "000", # 0
    "001", # 1
    "010", # 2
    "011", # 3
    "100", # 4
    "110" # 5
    ]
    #opt.weights = 'best.pt'
    #opt.source = "0"
    #opt.source = "./001-AltayMirzaliyev-1.mp4"
    #opt.source = "./20200224002.mp4"
    #opt.img_size = 416
    #opt.conf = 0.4
    #opt.save_txt = True
    #opt.view_img = True
    #opt.save_conf = True
    #opt.cache = True
    #opt.drv_gaze = True
    if opt.source == "nvargus":
        opt.source = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, framerate=(fraction)30/1 ! nvvidconv flip-method=5 ! video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
        #opt.source = "nvarguscamerasrc sensor-id=0"
    
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.DetectorWeights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.DetectorWeights)
        else:
            detect()
