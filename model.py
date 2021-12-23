import torch
import torch.nn as nn
import torch.nn.functional as F



class Block_Down(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(Block_Down,self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, (3,3))
        self.conv2 = nn.Conv2d(out_ch,out_ch, (3,3))
        self.drop = nn.Dropout(p=0.7)
    def forward_d(self,x):  
        layer1 = F.relu(self.conv2(F.relu(self.conv1(x))))
        return layer1
    def down_size(self,x):
        layer1 = F.max_pool2d(x, (2, 2))
        layer2 = self.drop(layer1)
        return layer2

class Block_Up(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(Block_Up,self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, (3,3))
        self.conv2 = nn.Conv2d(out_ch,out_ch, (3,3))
        self.convT = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)


    def forward_u(self,x):        
        layer1 = F.relu(self.conv2(F.relu(self.conv1(x))))
        return layer1

    def up_size(self,x):
        layer1 = self.convT(x)
        return layer1
    def merge(self,x,y):
        a1,b1,c1,d1 = x.size()
        a2,b2,c2,d2 = y.size()

        l1 = int((c2 -c1)/2)
        l2 = int((d2-d1)/2)
        # print(c1,d1,c2,d2)

        cropped_layer_y = y[:,:,l1:c2-l1,l2:d2-l2]
        # print(x.size(),cropped_layer_x.size(),y.size())
        layer_1 = torch.cat((cropped_layer_y,x), 1)
        return layer_1



class Logit(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(Logit,self).__init__()
        self.conv1 = nn.Conv2d(in_ch,out_ch, (1,1))

    def forward(self,x):
        layer_1 = torch.relu(self.conv1(x))      
        layer_2 = torch.softmax(layer_1,dim=1)
        return layer_2
    




class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.B1 = Block_Down(3,64)
        self.B2 = Block_Down(64,128)
        self.B3 = Block_Down(128,256)
        self.B4 = Block_Down(256,512)
        self.B5 = Block_Down(512,1024)
        
        self.B6 = Block_Up(1024,512)
        self.B7 = Block_Up(512,256)
        self.B8 = Block_Up(256,128)
        self.B9 = Block_Up(128,64)
        self.L = Logit(64,2)
        

    def forward(self,x):
    ##### Encoder
        
        layer_1 = self.B1.forward_d(x)
        layer_2 = self.B1.down_size(layer_1)        
        
        layer_3 = self.B2.forward_d(layer_2)
        layer_4 = self.B1.down_size(layer_3)

        layer_5 = self.B3.forward_d(layer_4)
        layer_6 = self.B3.down_size(layer_5)
        
        layer_7 = self.B4.forward_d(layer_6)
        layer_8 = self.B4.down_size(layer_7)
        
        layer_9 = self.B5.forward_d(layer_8)






    ###### Decoder 
                
        layer_10 = self.B6.up_size(layer_9)
        layer_11 = self.B6.merge(layer_10,layer_7)
        layer_12 = self.B6.forward_u(layer_11)
                      
        layer_13 = self.B7.up_size(layer_12)        
        layer_14 = self.B7.merge(layer_13,layer_5)
        layer_15 = self.B7.forward_u(layer_14)
                
        layer_16 = self.B8.up_size(layer_15)        
        layer_17 = self.B8.merge(layer_16,layer_3)
        layer_18 = self.B8.forward_u(layer_17)
                
        layer_19 = self.B9.up_size(layer_18)        
        layer_20 = self.B9.merge(layer_19,layer_1)
        layer_21 = self.B9.forward_u(layer_20)

        layer_22 = self.L.forward(layer_21)        

        return layer_22

