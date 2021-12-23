import torch
from torchvision import datasets, transforms
import argparse
import torch.optim as optim

from model import *
from dataset import *






########################################################


##### Parameter Initialization

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=3,type=int, help='size of the batch for loading the data')
parser.add_argument('--data_dir', default='./data',type=str, help='address of the data folder')
parser.add_argument('--epoch', default=1000,type=int, help='number of epochs for training ')
opt = parser.parse_args()

batch_size = opt.batch_size
data_dir = opt.data_dir
epoch = opt.epoch


#######################################################
###### LOAD DATA
def test_train_dataset(data_dir):
    image_path = data_dir + '/train/image/'
    image_label_path = data_dir + '/train/label/'
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
    # print(len(train_data))
    # train_set, val_set = torch.utils.data.random_split(train_data, [5, 25])
    test_data = datasets.ImageFolder(data_dir + '/test/data/', transform=test_transforms)

    # print(len(val_set))

    return train_data, test_data



##########################################################





def train(batch_size,epoch, train_data,save_model_path): 


    trainloader = torch.utils.data.DataLoader(train_data,batch_size,shuffle=True)

    # # ##########################################################
    # # ####  Training 
    torch.cuda.empty_cache()
    # torch.cuda.seed()
    model = UNet()
    model.to(device)
    torch.cuda.manual_seed(94)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-7)
    weights = [1, 3]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion =  nn.MSELoss()
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()


    


    for e in range(epoch):


        Loss = 0
        for sample in trainloader:

            optimizer.zero_grad()

            images =sample['image_x']
            b,d,r,c = images.size()
            label_size = int(((((((((((((((((r-4)/2)-4)/2)-4)/2)-4)/2)-4)*2)-4)*2)-4)*2)-4)*2)-4) 
            r1 = int((r-label_size)/2)
            labels = sample['image_y'][:,:,r1:r-r1,r1:c-r1]
            images = images.to(device)
            labels = (labels.to(device)).type(torch.cuda.LongTensor)
            
            out = model.forward(images)
            

            loss = criterion(out[:,:,:,:], labels[:,0,:,:])
            loss.backward()
            optimizer.step()
            Loss = Loss +loss.item()/b
        print("epoch=", e, "loss=",Loss)


            

        EPOCH = e        
        LOSS = loss.item()
        PATH = save_model_path+str(e)+".pt"
        torch.save({
                    'epoch': EPOCH,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': LOSS,
                    }, PATH)






# #  ####################################################################################
# #  #### Data Visualization     and TEST
def test(selected_model_epoch, test_data, save_model_path,image_test_path = data_dir+'/test/result/'):
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1,shuffle=False)


    epoch = selected_model_epoch
    PATH = save_model_path+str(epoch-1)+".pt"
    model = UNet()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

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

        # Out_img = out.to('cpu')[0,1,:,:]
        # print(Out_img)
        img_out = (out > t).float() *1
        # print(img_out.size())
        img_out = img_out[0,0,:,:]
        # print(img_out)
        img_out = img_out.to('cpu')
        trans = transforms.ToPILImage()
        trans(1-img_out-0.001).save(image_test_path+str(count)+'.jpg')
        trans(img[0,:,:,:]).save(image_test_path+'_main_'+str(count)+'.jpg')
        count = count + 1


if __name__ == "__main__":

    
    save_model_path = "./model/model_"
    selected_model_epoch = 600
    image_test_path = data_dir+'/test/result/'

    train_data, test_data = test_train_dataset(data_dir)
    # train(batch_size,epoch,train_data,save_model_path)

    test(selected_model_epoch,test_data, save_model_path,image_test_path)
