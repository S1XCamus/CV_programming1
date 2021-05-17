import os
import torch
from PIL import Image
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import models, transforms

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))


class Test_Dataset(Dataset):
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
        return img, img_name



if __name__ == '__main__':
    # environment
    use_gpu = torch.cuda.is_available()
    # parameters
    MAX_EPOCH = 20
    NUM_LABELS = 80
    BATCH_SIZE = 20
    # load train & val data
    image_datasets = {'test': Test_Dataset(img_path='dataset/test', txt_path='dataset/val_anno.txt',
                                           data_transform=data_transforms['test'])}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=False) for x in
                   ['test']}
    # reload model
    model = models.resnext101_32x8d(True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_LABELS)
    model.load_state_dict(torch.load('/content/drive/MyDrive/Colab Notebooks/model_rn101_aug1.pth'))
    if use_gpu:
        model = model.cuda()
    model.eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.9)
    # train & validate
    # load test data & output result
    f = open('181250122.txt', 'w')
    with torch.no_grad():
      for data in dataloaders['test']:
        # load data
        inputs, img_names = data
        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
        optimizer.zero_grad()  # 将网络中的所有梯度置0
        # forward progressing
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)  # 得到预测类别
        labels = preds.cpu().numpy().tolist()
        for i in range(BATCH_SIZE):
          name = img_names[i].strip().split('/')[-1]
          f.write(name + ' ' + str(labels[i]) + '\n')
    f.close()
