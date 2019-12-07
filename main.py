import torch
from torch import nn
import os
import argparse
import torch.utils.data as data
from torch import optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import tqdm
import numpy as np
import random
from tensorboardX import SummaryWriter
from datetime import datetime
import dateutil.tz


def trainer(opt):
    path = "./kills_per_state_per_month.csv"
    logger_path = mk_log_dir("./Logs/", opt)
    writer = SummaryWriter(logger_path['log_path'])
    print("loading")
    transform = transforms.Compose([
            #transforms.RandomCrop(crop_size, padding=padding),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            #transforms.Normalize((0.4914, 0.4822, 0.4465),
            #                     (0.2023, 0.1994, 0.2010)),
        ])
    train_data = train_data_sequence(path, transform)
    test_data = test_data_sequence(path, transform)
    print(len(train_data))
    print(len(test_data))
    train_dataloader = torch.utils.data.DataLoader(
                            train_data,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.n_threads,
                            pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(
                            test_data,
                            batch_size=90,
                            shuffle=True,
                            num_workers=opt.n_threads,
                            pin_memory=True)
    gpu_ids = [i for i in range(len(opt.gpu_ids.split(",")))]
    model = TemporalModel()
    model = nn.DataParallel(model.to("cuda:0"), device_ids=gpu_ids)
    model = model.to("cuda:0")
    optimizer = optim.SGD(
            model.parameters(),
            lr=opt.learning_rate,
            momentum=0.9,
            weight_decay=opt.weight_decay
            )
    loss = nn.CrossEntropyLoss()
    for epoch in range(0, 1000):
        train_loss = training(model, optimizer, loss, train_dataloader, epoch, writer)
        test_loss = testing(model, optimizer, loss, test_dataloader, epoch, writer)
        update_learning_rate(optimizer, opt.learning_rate, epoch)

def training(model, optimizer, Loss, data_loader, epoch, writer):
    model.train()
    losses = AverageMeter()
    for i, (inputs, state, month, targets) in enumerate(data_loader):
        inputs = Variable(inputs.cuda())
        targets = Variable(targets.cuda())
        state = Variable(state.cuda())
        month = Variable(month.cuda())
        outputs = model(inputs, state, month)
        loss = Loss(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(inputs[0],state[0], month[0], torch.max(outputs[0], 0)[1].item(), targets[0].item())
    print("Epoch: " + str(epoch) + ", avg loss: " + str(losses.avg))
    writer.add_scalar("loss/train", losses.avg, epoch)
    return losses.avg


def testing(model, optimizer, Loss, data_loader, epoch, writer):
    model.eval()
    losses = AverageMeter()
    for i, (inputs, state, month, targets) in enumerate(data_loader):
        inputs = Variable(inputs.cuda())
        targets = Variable(targets.cuda())
        state = Variable(state.cuda())
        month = Variable(month.cuda())
        outputs = model(inputs, state, month)
        loss = Loss(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
    print("Test, " + "Epoch: " + str(epoch) + ", avg loss: " + str(losses.avg))
    writer.add_scalar("loss/test", losses.avg, epoch)
    return losses.avg

class TemporalModel(nn.Module):
    def __init__(self):
        super(TemporalModel, self).__init__()
        
        #self.fc = nn.Linear(3, )
        self.h_size = 1000
        self.layer = 3
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.h_size, num_layers=self.layer, batch_first=True)
        #self.embedding_1 = nn.Linear(2, 3*self.h_size)
        self.embedding_2 = nn.Linear(3*self.h_size,200)
    def forward(self, x, s, m):
        x = x.unsqueeze(2)
        # x = self.embedding(x)
        h0 = Variable(torch.randn(self.layer, x.size(0), self.h_size).cuda())
        c0 = Variable(torch.randn(self.layer, x.size(0), self.h_size).cuda())
        #output = self.fc(x)
        output, _ = self.lstm(x, (h0, c0))
        output = output.contiguous().view(output.size(0), 3*self.h_size)
        #condition = torch.cat((s.unsqueeze(1), m.unsqueeze(1)), dim=1)
        #condition_feature = self.embedding_1(condition)
        #output = torch.cat((output, s.unsqueeze(1), m.unsqueeze(1)), dim=1)
        output = self.embedding_2(output)
        return output

class train_data_sequence(data.Dataset):
    def __init__(self, path, transform):
        self.path = path
        data_file = open(path, "r")
        self.data = []
        for line in data_file.readlines():
            line = line.replace("\n", "")
            line = [int(i) for i in line.split(",")]
            self.data.append(line)
        #print(self.data[0:5])
        self.transform = transform

    def __getitem__(self, index):
        data = self.data[index]
        killed_list = data[3:6]
        rand_num = int(6*random.random())
        for i in range(len(killed_list)):
            if killed_list[i] > 10 and data[6] > 10:
                killed_list[i] = int(max(killed_list[i] + rand_num-3, 0))
        killed_list = np.array(killed_list).astype(np.float32)
        killed_num = torch.from_numpy(killed_list)
        if data[6] > 10:
            target = torch.tensor(int(data[6])+ rand_num-3)
        else:
            target = torch.tensor(int(data[6]))
        state = torch.tensor(float(data[0]))
        month = torch.tensor(float(data[1]))
        return killed_num, state, month, target

    def __len__(self):
        return len(self.data)

class test_data_sequence(data.Dataset):
    def __init__(self, path, transform):
        self.path = path
        data_file = open(path, "r")
        self.data = []
        for line in data_file.readlines():
            line = line.replace("\n", "")
            line = [int(i) for i in line.split(",")]
            if line[7] > 10:
                self.data.append(line[0:2]+line[4:8])
            if line[2] > 10:
                self.data.append(line[0:6])
        #print(self.data[0:5])
        self.transform = transform

    def __getitem__(self, index):
        data = self.data[index]
        killed_list = data[2:5]
        killed_list = np.array(killed_list).astype(np.float32)
        killed_num = torch.from_numpy(killed_list)
        target = torch.tensor(int(data[5]))
        state = torch.tensor(float(data[0]))
        month = torch.tensor(float(data[1]))
        return killed_num, state, month, target

    def __len__(self):
        return len(self.data)

def update_learning_rate(optimizer, lr, epoch):
    if epoch > 200:
        lr = lr/10.
    if epoch > 400:
        lr = lr/100.
    if epoch > 600:
        lr = lr/1000.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mk_log_dir(save_root, opt):
    path_dict = {}
    save_root = save_root
    os.makedirs(save_root, exist_ok=True)
    exp_name = os.path.join(save_root, opt.exp_name)
    # set log path
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_name + '_' + timestamp
    os.makedirs(prefix, exist_ok=True)
    path_dict['prefix'] = prefix

    # set checkpoint path
    model_restore_path = os.path.join(prefix, 'Model')
    os.makedirs(model_restore_path, exist_ok=True)
    path_dict['model_restore_path'] = model_restore_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path, exist_ok=True)
    path_dict['log_path'] = log_path

    # # set sample image path for fid calculation
    # sample_path = os.path.join(prefix, 'Samples')
    # os.makedirs(sample_path, exist_ok=True)
    # path_dict['sample_path'] = sample_path

    return path_dict

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument(
        '--port', default=23412, type=int, help='Manually set random seed')
    parser.add_argument(
        '--learning_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument(
        '--weight_decay', default=5e-3, type=float, help='learning rate')
    parser.add_argument(
        '--exp_name', required=True, type=str, help='learning rate')
    parser.add_argument(
        '--n_threads', default=8, type=int, help='num_workers')
    parser.add_argument(
        '--batch_size', default=16, type=int, help='Manually set random seed')
    parser.add_argument(
        '--gpu_ids', default="0", type=str, help='gpus id')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    opt = parse_opts()
    torch.manual_seed(12345)
    torch.cuda.manual_seed(12345)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= opt.gpu_ids
    # os.environ['MASTER_PORT'] = opt.port
    trainer(opt)
