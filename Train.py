import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, models
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd


# 数据增强
data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(30),
        # transforms.RandomApply([
        #     transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.2),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # transforms.RandomSizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


# 加载val需要下面的内容
def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))


class Val_Dataset(Dataset):
    def __init__(self, img_path, txt_path, data_transform=None, loader=default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.strip().split(' ')[0]) for line in lines]
            self.img_label = [int(line.strip().split(' ')[-1]) for line in lines]
        self.data_transform = data_transform
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)
        if self.data_transform is not None:
            img = self.data_transform(img)
        return img, label


class Train_Dataset(Dataset):
    def __init__(self, img_path, data_transform=None, loader=default_loader):
        self.img_name = []
        self.img_label = []
        for rt, dirs, files in os.walk(img_path):
            for f in files:
                full_name = os.path.join(rt, f)
                self.img_name.append(full_name)
                self.img_label.append(int(rt.split('/')[-1]))
        self.data_transform = data_transform
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)
        if self.data_transform is not None:
            img = self.data_transform(img)
        return img, label


if __name__ == '__main__':
    # environment
    use_gpu = torch.cuda.is_available()
    # use_gpu = False
    f = open('result_rn101_aug1.txt', 'w')
    # parameters
    MAX_EPOCH = 20
    NUM_LABELS = 80
    INPUTS_NUM = {'train': 20000, 'val': 10000}
    BATCH_SIZE = 20
    # load train & val data
    image_datasets = {'train': Train_Dataset(img_path='dataset/train', data_transform=data_transforms['train']),
                      'val': Val_Dataset(img_path='dataset/val', txt_path='dataset/val_anno.txt', data_transform=data_transforms['val'])}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True) for x in
                   ['train', 'val']}
    # model
    # model = models.wide_resnet101_2(True)
    # model = models.wide_resnet50_2(True)
    model = models.resnext101_32x8d(True)
    # model = models.resnet18(True)
    # model = models.resnext50_32x4d(True)
    # model = models.resnet101(True)
    # model = models.resnet50(True)
    # model = models.inception_v3(True)
    # model = models.densenet161(True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_LABELS)
    if use_gpu:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.9)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # train & validate
    last_val_acc = 0
    last_model_wts = model.state_dict()
    # best_val_acc = 0
    # best_model_wts = model.state_dict()
    stop = False
    for e in range(MAX_EPOCH):
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()  # 更新学习率
                model.train()
            else:
                model.eval()
            tmp_loss = 0
            tmp_acc = 0
            for data in dataloaders[phase]:
                # load data
                inputs, labels = data
                #         print(inputs.size())
                #         break
                #     break
                # break
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()  # 将网络中的所有梯度置0
                # forward progressing
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs.data, 1)  # 得到预测类别
                # backward progressing
                if phase == 'train':
                    loss.backward()  # 回传损失
                    optimizer.step()  # 更新参数
                # calculate loss and accuracy
                tmp_loss += loss.item()
                tmp_acc += torch.sum(preds == labels.data).to(torch.float32)
                # break###############
            tmp_loss /= INPUTS_NUM[phase]
            tmp_acc /= INPUTS_NUM[phase]
            print('{} Loss: {:.3f} Acc: {:.3f}'.format(phase, tmp_loss, tmp_acc))
            f.write('{} Loss: {:.3f} Acc: {:.3f}\n'.format(phase, tmp_loss, tmp_acc))
            # record model
            # if phase == 'val' and tmp_acc > best_val_acc:
            #     best_val_acc = tmp_acc
            #     best_model_wts = model.state_dict()
            if phase == 'val':
                # 早停：验证集准确率下降时停止
                if tmp_acc <= last_val_acc:
                    stop = True
                else:
                    last_model_wts = model.state_dict()
                    last_val_acc = tmp_acc
            #break###############
        if stop:
            # store the model
            torch.save(last_model_wts, 'model_rn101_aug1.pth')
            f.write('Final acc: {:.3f}\n'.format(last_val_acc))
            break
    f.close()
    # torch.save(model.state_dict(), 'model.pth')
