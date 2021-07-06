
#######
###  Check softmax
### check cross validation


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import argparse
import os
from PIL import Image
import torch.optim as optim

class Block_Down(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(Block_Down,self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, (3,3))
        self.conv2 = nn.Conv2d(out_ch,out_ch, (3,3))
        self.drop = nn.Dropout(p=0.5)
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









class BioDataset(Dataset):
    
    def __init__(self,input_path,seg_images_path,transform_image=None):
        '''
        
        input_path = path to input image folders
        seg_images_path = path to segmentation maps class folder
        '''
        self.input_path = input_path
        self.seg_images_path = seg_images_path
        self.transform_image = transform_image
        
        
    def getImagesNames(self,path):
      
        
        images_names = os.listdir(self.input_path)        
        return images_names
    
    def __len__(self):
        return len( os.listdir(self.input_path))
    
    def __getitem__(self, idx):
        file_names = self.getImagesNames(self.input_path)
        file_path_x = self.input_path+file_names[idx]
        file_path_y = self.seg_images_path+file_names[idx]
        
        img = Image.open(file_path_x).convert('RGB')
        label = Image.open(file_path_y)
        
        I1 = transforms.ToTensor()(img)
        I2 = transforms.ToTensor()(label)

        merged_I = torch.cat((I1,I2),dim = 0 )

        sample = {'image_x':img,'image_y':label}
        PIL_merged_I = transforms.ToPILImage()(merged_I)
        if self.transform_image:
            

            tensor_merged_I = self.transform_image(PIL_merged_I)
            sample['image_x'] = tensor_merged_I[0:3,:,:] 
            sample['image_y'] =  torch.unsqueeze(tensor_merged_I[3,:,:],0)
            
    
        return sample

 

########################################################


##### Parameter Initialization

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=3,type=int, help='size of the batch for loading the data')
parser.add_argument('--data_dir', default='./data',type=str, help='address of the data folder')
parser.add_argument('--epoch', default=5000,type=int, help='number of epochs for training ')
opt = parser.parse_args()

batch_size = opt.batch_size
data_dir = opt.data_dir
epoch = opt.epoch
image_path = data_dir + '/train/image/'
image_label_path = data_dir + '/train/label/'
image_test_path = data_dir+'/test/result/'

#######################################################
###### LOAD DATA

train_transforms = transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.Resize(572),
                                transforms.RandomRotation(30),
                                transforms.RandomRotation(15),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()])

test_transforms = transforms.Compose([transforms.Resize(572),
                                      transforms.ToTensor()])



train_data = BioDataset(image_path, image_label_path, transform_image= train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test/data/', transform=test_transforms)



trainloader = torch.utils.data.DataLoader(train_data,batch_size=3,shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=1)

##########################################################







# # # ##########################################################
# # # ####  Training 
# torch.cuda.empty_cache()
# # torch.cuda.seed()
# model = UNet()
# model.to(device)
# torch.cuda.manual_seed(94)
# model.train()

# weights = [1, 3]
# class_weights = torch.FloatTensor(weights).cuda()
# criterion = nn.CrossEntropyLoss(weight=class_weights)
# # criterion =  nn.MSELoss()
# # criterion = nn.BCEWithLogitsLoss()
# # criterion = nn.CrossEntropyLoss()


# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# # optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-7)


# for e in range(epoch):


#     Loss = 0
#     for sample in trainloader:

#         optimizer.zero_grad()

#         images =sample['image_x']
#         b,d,r,c = images.size()
#         label_size = int(((((((((((((((((r-4)/2)-4)/2)-4)/2)-4)/2)-4)*2)-4)*2)-4)*2)-4)*2)-4) 
#         r1 = int((r-label_size)/2)
#         labels = sample['image_y'][:,:,r1:r-r1,r1:c-r1]
#         images = images.to(device)
#         labels = (labels.to(device)).type(torch.cuda.LongTensor)
        
#         out = model.forward(images)
        

#         loss = criterion(out[:,:,:,:], labels[:,0,:,:])
#         loss.backward()
#         optimizer.step()
#         Loss = Loss +loss.item()/b
#     print("epoch=", e, "loss=",Loss)


           

#     EPOCH = e
#     PATH = "./model/model_"+str(e)+".pt"
#     LOSS = loss.item()

#     torch.save({
#                 'epoch': EPOCH,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': LOSS,
#                 }, PATH)






# #  ####################################################################################
# #  #### Data Visualization     and TEST
epoch = 450
PATH = "./model/model_"+str(epoch-1)+".pt"
model = UNet()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
model.to(device)
count = 0


for sample in testloader:
    images =sample[0]
    t = 0.034
    b,d,r,c = images.size()
    # print(b,d,r,c)
    label_size = int(((((((((((((((((r-4)/2)-4)/2)-4)/2)-4)/2)-4)*2)-4)*2)-4)*2)-4)*2)-4) 
    r1 = int((r-label_size)/2)
    img = images.to(device)
    
    out = model.forward(img) 
    img = img.to('cpu')

    print(out[0,0,:,:])
    # Out_img = out.to('cpu')[0,1,:,:]
    # print(Out_img)
    img_out = (out > t).float() *1
    # print(img_out.size())
    img_out = img_out[0,0,:,:]
    # print(img_out)
    img_out = img_out.to('cpu')
    trans = transforms.ToPILImage()
    trans(1-img_out).save(image_test_path+str(count)+'.jpg')
    trans(img[0,:,:,:]).save(image_test_path+'_main_'+str(count)+'.jpg')
    count = count + 1




