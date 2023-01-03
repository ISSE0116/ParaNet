################################
#                              #
#          regression          #
#           Training           #
################################
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import math
import datetime
import sys

plt.ion()

#training img dir
data_dir = '/mnt/data1/kikuchi/kikuchisan/reg'

#####define augment#####
batch_size = int(sys.argv[1]) 
num_epochs = int(sys.argv[2])
lr = float(sys.argv[3])
cuda_num = sys.argv[4]

########parameter#######
step_size = int(num_epochs * 0.9)
wd = 0.0001 
########################

#lossとaccの遷移を記録、グラフに使う
loss_t=[]
loss_v=[]
acc_t=[]
acc_v=[]

#########################data augmentation(old)###########################

image_datasets = datasets.ImageFolder(data_dir,data_transforms['train'])

#devide to train data and validation data
train_size = int(0.8 * len(image_datasets))
val_size = len(image_datasets) - train_size
train, val = torch.utils.data.random_split(image_datasets, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size, shuffle=True)
data_loader = {'train':train_loader,'val':val_loader}

train_sizes = len(train)
val_sizes = len(val)
dataset_sizes = {'train':train_sizes,'val':val_sizes}

class_names = image_datasets.classes
device = torch.device("cuda:{}".format(cuda_num) if torch.cuda.is_available() else "cpu")

##########################about training#############################

print("\n\n\n")
print("[ABOUT_regression]")
print("\033[34m###################################################################\033[0m")
print(f"\033[34mfull: {len(image_datasets)} -> train: {len(train)}, validation: {len(val)}\033[0m")
print("\033[34m###################################################################\033[0m")
print("\033[34mlr(Initial): {}, batch_size: {}, num_epoch: {}, step_size: {}, weight_decay: {}\033[0m".format(lr, batch_size, num_epochs, step_size, wd))
print("\033[34m###################################################################\033[0m")
print("\n\n\n")


#########################data augmentation(new!!!!!)############################

#can generate image and label at the same time.
def trainAug()
    transform = {
        transform.Compose([
            transforms.RandomAffine(translate = (0, 0.5))
    ])
    
    image_datagen = ImageFolder("path", transform) 

#########################data augmentation(new2!!!!!)###########################

"""
class MyTransforms:
    def __init__(self) -> None:
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.from_numpy(x.astype(np.float32))
        return x
"""

class MyDataset(Dataset):
    def __init__(self, root: str, transforms) -> None:
        '''super().__init__()'''
        self.data = Path(root).glob("/.json")
        self.df = self.create_csv(path)
        self.data_transform = data_transform

    def create_csv(self, path):
        image_path = []  #create a list to save image_path

        for file_name in glob(path + '/*.png'):
            basename = os.path.basename(file_name)
            image_path.append(basename)

        df = pd.DataFrame(image_path, columns=['path']) 

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:  #データ取り出し時に呼び出し、indexがlen()で取得した長さだけ繰り返される
        data = self.data[index]["data"]
        label = self.data[index]["label"]

        #dataの変形
        data = self.transforms(data)

        return data, label

    def __len__(self) -> int:   #len()を使用するときに呼び出される関数
        return len(self.data)

transforms = transforms.Compose([
    transfors.ToTensor(),
    MyTransforms()
])

dataset = MyDataset(
    # DO: fix to your original
    root="path to dataset's root",
    transforms=transforms
)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=64,
    shuffle=True
)

##############################dataset###############################

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):  #初期化
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self): #データの個数を返す
        return len(self.img_labels)

    def __getitem__(self, idx): #__len()__で返却された数値分のiが処理される for i in range()のような感じ
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

##########################training models##############################

best_acc = 0.0
dt_now = 0
#Get a batch of Graining data
inputs, classes = next(iter(train))
#Make a grid from batch
out = torchvision.utils.make_grid(inputs)

def train_model(model, criterin, optimizer, scheduler, num_epochs):
    since = time.time()
    i = 1
    best_models_wts = copy.deepcopy(model.state_dict())
    global loss_t, loss_v, acc_t, acc_v, best_acc, dt_now

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
               model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in data_loader[phase]:          #dataloaderからdataset呼び出し
                inputs = inputs.to(device)                     #GPUに転送
                labels = labels.to(device) 
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'): #train時のみ勾配の算出をオンにするの意
                    #m = nn.dropout(inputs)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    i += 1

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)   #lossとbatchサイズとの掛け算
                running_corrects += torch.sum(preds == labels.data)
            
            if(phase == 'train'):
                epoch_loss_t = running_loss / dataset_sizes['train']
                epoch_acc_t = running_corrects.double() / dataset_sizes['train']
                loss_t.append(epoch_loss_t)
                acc_t.append(epoch_acc_t.item())
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss_t, epoch_acc_t))
            else:
                epoch_loss_v = running_loss / dataset_sizes['val']
                epoch_acc_v = running_corrects.double() / dataset_sizes['val']
                loss_v.append(epoch_loss_v)
                acc_v.append(epoch_acc_v.item())
                print('{} Loss: {:.4f} '.format(phase, epoch_loss_v), end = '')

                if epoch_acc_v > best_acc:
                    best_acc = epoch_acc_v
                    best_models_wts = copy.deepcopy(model.state_dict())
                    print('\033[32mAcc: {:.4f}\033[0m'.format(epoch_acc_v))
                else:
                    print('Acc: {:.4f}'.format(epoch_acc_v))

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print('Best val Acc: {:4f}'.format(best_acc))
    
    dt_now = datetime.datetime.now()
    dt_now = str(dt_now.month) + str(dt_now.day) + '-' + str(dt_now.hour) + str(dt_now.minute) 
    model_path = 'model_path_' + '{}-{}-{}_'.format(lr, batch_size, num_epochs) + dt_now
    torch.save(best_models_wts, os.path.join('../weight_path/weight_path_resnet18', model_path))
    print()
    print('!!!!!save_{}!!!!!'.format(model_path))
    return model

###################################define regression model##################################
#define model
model_reg = models.resnet18(pretrained=True)
#number of head 
num_ftrs = model_reg.fc.in_features
#change to regression
model_reg.fc = nn.Linear(num_ftrs, 1)
#send model to GPU
model_reg = model_reg.to(device)        
#define criterion
criterion = nn.MSELoss()          
#define optimizer
optimizer = optim.SGD(model_ft.parameters(), lr, momentum=0.9, weight_decay=wd)     
#define schedular
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size, gamma=0.1)           
#instance train_model
train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs)

####################################plot the result graph###################################
fig = plt.figure(figsize = (10,5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.plot(range(len(loss_t)), loss_t, label="Loss(Train)")
ax1.plot(range(len(loss_v)), loss_v, label="Loss(Val)")
ax2.plot(range(len(acc_t)), acc_t, label="Acc(Train)")
ax2.plot(range(len(acc_v)), acc_v, label="Acc(Val)")
ax1.legend()
ax2.legend()
plt.title('size[ [train]:{}  [val]:{} ]  [lr]:{}  [batch_size]:{}  [epoch]:{}  [best_acc_val]:{:.3f}'.format(train_sizes, val_sizes, lr, batch_size, num_epochs, best_acc), loc = 'right', y = 1.05, weight = 1000, color = 'green')

ax1.set_ylim(0,7)
ax2.set_ylim(0,1)
ax1.set_yticks([0,1,2,3,4,5,6,7])
ax2.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
ax1.grid(which = "major", axis = "y", color = "black", alpha = 0.8, linestyle = "--", linewidth = 0.5)
ax2.grid(which = "major", axis = "y", color = "black", alpha = 0.8, linestyle = "--", linewidth = 0.5)
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Acc")
graph = 'train_result_graph_' + '{}-{}-{}_'.format(lr, batch_size, num_epochs) + dt_now + '_aug'  + '.png' 
plt.savefig(os.path.join("../graph/", graph))

print()
print("!!!!!end_to_plot_graph!!!!!")
print()
##############################################################################################
