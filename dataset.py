import torch
import os
from torch.utils.data import Dataset
from torchvision import  transforms
from PIL import Image


class BioDataset(Dataset):
    
    def __init__(self,input_path,seg_images_path,transform_image=None):
        '''
        
        input_path = path to input image folders
        seg_images_path = path to segmentation maps class folder
        '''
        self.input_path = input_path
        self.seg_images_path = seg_images_path
        self.transform_image = transform_image
        
        
    def getImagesNames(self):
      
        
        images_names = os.listdir(self.input_path)        
        return images_names
    
    def __len__(self):
        return len( os.listdir(self.input_path))
    
    def __getitem__(self, idx):
        file_names = self.getImagesNames()
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

 
