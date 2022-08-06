import torch
import torch.nn as nn
import pandas as pd

from utils.torch_utils import select_device
from utils.general import scale_coords, xyxy2xywh

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
    
class Lstm_decision():
    def __init__(self, model_path, device):
        self.device = select_device(device)
        n_hidden = 128
        n_joints = 19
        #n_categories = 6
        regression_out = 3
        n_layer = 3
        self.model = LSTM(n_joints,n_hidden,regression_out,n_layer)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
    
    def elestiem(self,pred,names,im0shape,imgshape):
        for i, det in enumerate(pred):
            if len(det)==0:
                df = pd.DataFrame(columns=['Class','Xc','Yc','W','H'])
            else:
                gn = torch.tensor(im0shape)[[1, 0, 1, 0]]
                det[:, :4] = scale_coords(imgshape, det[:, :4], im0shape).round()
                lst=[]
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (int(cls.tolist()), *xywh)
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
    def lstm_det(self,lstm_dat):
        lstm_dat = torch.as_tensor([[lstm_dat]])
        lstm_dat.to(self.device)
        output = self.model(lstm_dat)
        #top_n, top_i = output.topk(1)
        output = torch.tensor([min(1.0,max(0.0,elem.item())) for elem in output[0]]).to(self.device)
        o = [(1 if elem.item()>=0.5 else 0) for elem in output]
        return o