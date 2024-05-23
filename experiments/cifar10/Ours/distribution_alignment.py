'''
all clients have labels and train together, with samples all aligned.
'''
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms import transforms
from datasets.cifar10_server import *
from mmd import *
from torch.distributions import Normal, kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal

# p = Normal(0, 1)
# q = Normal(1, 2)
#
# kl = kl_divergence(p, q)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'

from datetime import datetime

currentDateAndTime = datetime.now()

print("The current date and time is", currentDateAndTime)
path = './checkpoint/best_cifar10.ckpt_' + str(currentDateAndTime)


# Output: The current date and time is 2022-03-19 10:05:39.482383


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1)

EPOCH = 70
BATCH_SIZE = 500
# LR = 0.001
LR = 0.01
DOWNLOAD_MNIST = True
# 0.1 is the best
LAMBDA = 0.1

'''
##### Params setting #####
Local: True False False
Vanilla VFL: False True False
Ours: True True True
Ablation with classifier: True True False
Ablation with MAE: False True True
'''

# Train classifier
TRAIN_CLASSIFIER = False
# Train server
TRAIN_SERVER = True

# augmentation
print('==> Preparing data..')
# data process
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_data = torchvision.datasets.CIFAR10(
    root='/data0/data_sw/',
    train=True,
    transform=transform,
    download=True
)

train_data, val_data = Data.random_split(train_data, [45000, 5000])

'''
Divide dataset according to aligned samples
'''
# # aligned samples 500
train_data_aligned, train_data_AB = Data.random_split(train_data, [500, 44500])
train_data_A, train_data_B = Data.random_split(train_data_AB, [22250, 22250])

# # aligned samples 1000
# train_data_aligned, train_data_AB = Data.random_split(train_data, [1000, 44000])
# train_data_A, train_data_B = Data.random_split(train_data_AB, [22000, 22000])

# aligned samples 2000
# train_data_aligned, train_data_AB = Data.random_split(train_data, [2000, 43000])
# train_data_A, train_data_B = Data.random_split(train_data_AB, [21500, 21500])

# aligned samples 4000
# train_data_aligned, train_data_AB = Data.random_split(train_data, [4000, 41000])
# train_data_A, train_data_B = Data.random_split(train_data_AB, [20500, 20500])

# # aligned samples 8000
# train_data_aligned, train_data_AB = Data.random_split(train_data, [8000, 37000])
# train_data_A, train_data_B = Data.random_split(train_data_AB, [18500, 18500])

# # aligned samples 10000
# train_data_aligned, train_data_AB = Data.random_split(train_data, [10000, 35000])
# train_data_A, train_data_B = Data.random_split(train_data_AB, [17500, 17500])

# # all aligned
# train_data_aligned, train_data_AB = Data.random_split(train_data, [45000, 0])
# train_data_A, train_data_B = Data.random_split(train_data_AB, [0, 0])

# train_data_server = CustomDataset(train_data, 500)

train_data_A_with_aligned = Data.ConcatDataset([train_data_A, train_data_aligned])

train_data_B_with_aligned = Data.ConcatDataset([train_data_B, train_data_aligned])

# test dataset
test_data = torchvision.datasets.CIFAR10(
    root='/data0/data_sw/',
    train=False,
    transform=transform
)
# aligned dataset
train_aligned_loader = Data.DataLoader(
    dataset=train_data_aligned,
    batch_size=BATCH_SIZE,
    shuffle=True
)
# dataset loader on client A
train_A_loader = Data.DataLoader(
    dataset=train_data_A_with_aligned,
    batch_size=BATCH_SIZE,
    shuffle=True
)
# dataset loader on client B
train_B_loader = Data.DataLoader(
    dataset=train_data_B_with_aligned,
    batch_size=BATCH_SIZE,
    shuffle=True
)
# validate set loader
val_loader = Data.DataLoader(
    dataset=val_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)
# test set loader
test_loader = Data.DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)
# loader on server for MAE estimating missing parts of B
train_A_MAE_loader = Data.DataLoader(
    dataset=train_data_A,
    batch_size=len(train_data_A),
    shuffle=True
)
# loader on server for MAE estimating missing parts of A
train_B_MAE_loader = Data.DataLoader(
    dataset=train_data_B,
    batch_size=len(train_data_B),
    shuffle=True
)


class VFL_Base(nn.Module):
    def __init__(self):
        super(VFL_Base, self).__init__()
        self.client_A = VFL_Client()
        self.client_B = VFL_Client()
        self.server = VFL_Server()

    def forward(self, x):
        x_client_A = x[:, :, :, :16]
        x_client_B = x[:, :, :, 16:]
        '''
        img = x_client_A[0]  
        img = img.numpy()  # FloatTensor to ndarray
        img = np.transpose(img, (1, 2, 0))  # dimension of channel to the last
        # show image
        plt.imshow(img)
        plt.show()
        '''
        x_client_A, result_A = self.client_A(x_client_A)
        x_client_B, result_B = self.client_B(x_client_B)
        result = self.server(torch.cat([x_client_A, x_client_B], dim=1))
        return result, result_A, result_B, x_client_A, x_client_B


class VFL_Server(nn.Module):
    def __init__(self):
        super(VFL_Server, self).__init__()
        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        result = self.classifier(x)
        return result


class VFL_Client(nn.Module):
    def __init__(self):
        super(VFL_Client, self).__init__()
        # padding = (kernel_size-1)/2
        # (3,32,16)
        self.client_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),  # (32,32,16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (32,16,8)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # (64,16,8)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # (64,16,8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (64,8,4)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # (128,8,4)
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # (128,8,4)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (128,4,2)
        )

        self.Linear = nn.Linear(128 * 4 * 2, 128)
        self.classifier = nn.Linear(128, 10)

    # forward
    def forward(self, x):
        x = self.client_conv(x)
        x = self.Linear(x.view(x.size(0), -1))
        result = self.classifier(x)
        return x, result


# test_x = torch.unsqueeze(test_data.train_data, dim=1).type(torch.FloatTensor)[:2000] / 255
# test_y = test_data.test_labels[:2000]
#
# test_x.cuda()
# test_y.cuda()


# Train
def train(model, epoch):
    model.train()
    # params for A
    train_loss_A = 0
    correct_A = 0
    total_A = 0
    best_acc_A = 0
    # params for B
    train_loss_B = 0
    correct_B = 0
    total_B = 0
    best_acc_B = 0
    # params for server
    train_loss_server = 0
    correct_server = 0
    total_server = 0

    best_acc_AB = 0
    best_acc_server = 0

    for ep in range(epoch):
        if TRAIN_CLASSIFIER:
            # train client A
            print('---------------------Train Client A---------------------')
            for batch_idx, (inputs, targets) in enumerate(train_A_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                # init grad as 0
                optimizer.zero_grad()
                result, result_A, result_B, _, _ = model(inputs)
                # calculate loss
                loss = loss_func(result_A, targets)
                # l2_lambda = 0.0005
                # l2_norm = sum(p.pow(2.0).sum()
                #               for p in model.parameters())
                #
                # loss = loss + l2_lambda * l2_norm
                # loss backward
                loss.backward()
                # update params
                optimizer.step()
                # calculate loss sum
                train_loss_A += loss.item()
                _, predicted = result_A.max(1)
                total_A += targets.size(0)
                correct_A += predicted.eq(targets).sum().item()

                # train_acc
                train_acc_A = correct_A / total_A

                if batch_idx % 20 == 0:
                    print(
                        'Epoch: {}, {}\{}: train loss_A: {:.4f}, accuracy_A: {:.4f}'.format(ep + 1, batch_idx + 1,
                                                                                            len(train_A_loader),
                                                                                            loss.item(), train_acc_A))
            # train client B
            print('---------------------Train Client B---------------------')
            for batch_idx, (inputs, targets) in enumerate(train_B_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                # init grad as 0
                optimizer.zero_grad()
                result, result_A, result_B, _, _ = model(inputs)
                # calculate loss
                loss = loss_func(result_B, targets)
                # l2_lambda = 0.0005
                # l2_norm = sum(p.pow(2.0).sum()
                #               for p in model.parameters())
                #
                # loss = loss + l2_lambda * l2_norm
                # loss backward
                loss.backward()
                # update params
                optimizer.step()
                # calculate loss sum
                train_loss_B += loss.item()
                _, predicted = result_B.max(1)
                total_B += targets.size(0)
                correct_B += predicted.eq(targets).sum().item()

                # train_acc
                train_acc_B = correct_B / total_B

                if batch_idx % 20 == 0:
                    print(
                        'Epoch: {}, {}\{}: train loss_B: {:.4f}, accuracy_B: {:.4f}'.format(ep + 1, batch_idx + 1,
                                                                                            len(train_B_loader),
                                                                                            loss.item(), train_acc_B))
        # train server
        if TRAIN_SERVER:
            print('---------------------Train Server---------------------')
            for batch_idx, (inputs, targets) in enumerate(train_aligned_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                # init grad as 0
                optimizer.zero_grad()
                result, result_A, result_B, x_client_A_aligned, x_client_B_aligned = model(inputs)
                # calculate loss
                loss = loss_func(result, targets)
                miu_S_A = torch.mean(x_client_A_aligned, dim=0)
                x_client_A_aligned_t = x_client_A_aligned - x_client_A_aligned.mean(0)
                sigma_S_A = x_client_A_aligned_t.t() @ x_client_A_aligned_t
                # sigma_S_A = torch.mean(x_client_A_aligned, dim=0)
                miu_S_B = torch.mean(x_client_B_aligned, dim=0)
                x_client_B_aligned_t = x_client_B_aligned - x_client_B_aligned.mean(0)
                sigma_S_B = x_client_B_aligned_t.t() @ x_client_B_aligned_t
                # sigma_S_B = torch.mean(x_client_B_aligned, dim=0)

                # align A
                dataloader_iterator = iter(train_A_loader)
                for p in range(5):
                    try:
                        (inputs_A, targets_A) = next(dataloader_iterator)
                    except StopIteration:
                        dataloader_iterator = iter(train_A_loader)
                        (inputs_A, targets_A) = next(dataloader_iterator)
                _, _, _, x_client_A_unaligned, _ = model(inputs_A)
                # calculate loss
                # loss_align_A = LAMBDA * mmd(x_client_A_aligned, x_client_A_unaligned)

                miu_A = torch.mean(x_client_A_unaligned, dim=0)
                x_client_A_unaligned_t = x_client_A_unaligned - x_client_A_unaligned.mean(0)
                sigma_A = x_client_A_unaligned_t.t() @ x_client_A_unaligned_t
                # sigma_A = torch.var(x_client_A_unaligned, dim=0)
                D1 = MultivariateNormal(miu_A, sigma_A)
                D2 = MultivariateNormal(miu_S_A, sigma_S_A)
                loss_align_A = LAMBDA * kl_divergence(D1, D2)

                # align B
                dataloader_iterator = iter(train_B_loader)
                for p in range(5):
                    try:
                        (inputs_B, targets_B) = next(dataloader_iterator)
                    except StopIteration:
                        dataloader_iterator = iter(train_B_loader)
                        (inputs_B, targets_B) = next(dataloader_iterator)
                _, _, _, _, x_client_B_unaligned = model(inputs_B)
                # calculate loss
                # loss_align_B = LAMBDA * mmd(x_client_B_aligned, x_client_B_unaligned)

                miu_B = torch.mean(x_client_B_unaligned, dim=0)
                x_client_B_unaligned_t = x_client_B_unaligned - x_client_B_unaligned.mean(0)
                sigma_B = x_client_B_unaligned_t.t() @ x_client_B_unaligned_t
                # sigma_B = torch.var(x_client_B_unaligned, dim=0)
                loss_align_B = LAMBDA * kl_divergence(MultivariateNormal(miu_B, sigma_B), MultivariateNormal(miu_S_B, sigma_S_B))


                loss = loss + loss_align_A + loss_align_B

                # inputs, targets = inputs.cuda(), targets.cuda()
                # # init grad as 0
                # optimizer.zero_grad()
                # result, result_A, result_B, _, _ = model(inputs)
                # # calculate loss
                # loss = loss_func(result, targets)
                # # l2_lambda = 0.0005
                # # l2_norm = sum(p.pow(2.0).sum()
                # #               for p in model.parameters())
                # #
                # # loss = loss + l2_lambda * l2_norm

                # loss backward
                loss.backward()
                # update params
                optimizer.step()

                # calculate loss sum
                train_loss_server += loss.item()
                _, predicted = result.max(1)
                total_server += targets.size(0)
                correct_server += predicted.eq(targets).sum().item()

                # train_acc
                train_acc_server = correct_server / total_server

                if batch_idx % 20 == 0:
                    print(
                        'Epoch: {}, {}\{}: train loss_server: {:.4f}, accuracy_server: {:.4f}'.format(ep + 1,
                                                                                                      batch_idx + 1,
                                                                                                      len(train_aligned_loader),
                                                                                                      loss.item(),
                                                                                                      train_acc_server))
        if TRAIN_CLASSIFIER and not TRAIN_SERVER:
            best_acc_server, best_acc_AB = val(model, ep, best_acc_AB, best_acc_server, classifier=True)
        else:
            best_acc_server, best_acc_AB = val(model, ep, best_acc_AB, best_acc_server)
        model.train()
        if (ep + 1) % 10 == 0:
            best_model = nn.DataParallel(VFL_Base()).cuda()
            best_model.load_state_dict(torch.load(path)['net'])
            test(best_model, ep)


def val(model, epoch, best_acc_AB=0, best_acc_server=0, classifier=False):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    # params for A
    val_loss_A = 0
    correct_A = 0
    total_A = 0
    # params for B
    val_loss_B = 0
    correct_B = 0
    total_B = 0
    # params for server
    val_loss_server = 0
    correct_server = 0
    total_server = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            result, result_A, result_B, _, _ = model(inputs)
            loss_A = loss_func(result_A, targets)
            loss_B = loss_func(result_A, targets)
            loss_server = loss_func(result_A, targets)

            val_loss_A += loss_A.item()
            _, predicted = result_A.max(1)
            total_A += targets.size(0)
            correct_A += predicted.eq(targets).sum().item()

            val_loss_B += loss_B.item()
            _, predicted = result_B.max(1)
            total_B += targets.size(0)
            correct_B += predicted.eq(targets).sum().item()

            val_loss_server += loss_server.item()
            _, predicted = result.max(1)
            total_server += targets.size(0)
            correct_server += predicted.eq(targets).sum().item()

        val_acc_A = correct_A / total_A
        val_acc_B = correct_B / total_B
        val_acc_server = correct_server / total_server
        print(
            'Epoch: {}, val accuracy_A: {:.4f}, val accuracy_B: {:.4f}, val accuracy_server: {:.4f}'.format(epoch + 1,
                                                                                                            val_acc_A,
                                                                                                            val_acc_B,
                                                                                                            val_acc_server))

    if classifier:
        acc_AB = 100. * (val_acc_A + val_acc_B) / 2
        if acc_AB > best_acc_AB:
            print('Save best classifiers..')
            state = {
                'net': model.state_dict(),
                'acc': acc_AB,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, path)
            model.load_state_dict(torch.load(path)['net'])
            best_acc_AB = acc_AB
        return best_acc_server, best_acc_AB
    else:
        acc_AB = 100. * 100. * (val_acc_A + val_acc_B) / 2
        acc_server = 100. * val_acc_server
        # if acc_server > best_acc_server and acc_AB > best_acc_AB:
        if acc_server > best_acc_server:
            print('Save best server..')
            state = {
                'net': model.state_dict(),
                'acc': acc_server,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, path)
            model.load_state_dict(torch.load(path)['net'])
            best_acc_AB = acc_AB
            best_acc_server = acc_server
        return best_acc_server, best_acc_AB


def test(model, epoch):
    model.eval()
    # params for A
    test_loss_A = 0
    correct_A = 0
    total_A = 0
    # params for B
    test_loss_B = 0
    correct_B = 0
    total_B = 0
    # params for server
    test_loss_server = 0
    correct_server = 0
    total_server = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            result, result_A, result_B, _, _ = model(inputs)
            loss_A = loss_func(result_A, targets)
            loss_B = loss_func(result_B, targets)
            loss = loss_func(result, targets)

            test_loss_A += loss_A.item()
            _, predicted = result_A.max(1)
            total_A += targets.size(0)
            correct_A += predicted.eq(targets).sum().item()

            test_loss_B += loss_B.item()
            _, predicted = result_B.max(1)
            total_B += targets.size(0)
            correct_B += predicted.eq(targets).sum().item()

            test_loss_server += loss.item()
            _, predicted = result.max(1)
            total_server += targets.size(0)
            correct_server += predicted.eq(targets).sum().item()

        test_acc_A = correct_A / total_A
        test_acc_B = correct_B / total_B
        test_acc_server = correct_server / total_server
        print(
            'Epoch: {}, test accuracy_A: {:.4f}, test accuracy_B: {:.4f}, test accuracy_server: {:.4f}'.format(
                epoch + 1, test_acc_A, test_acc_B, test_acc_server))


vfl_model = nn.DataParallel(VFL_Base()).cuda()
optimizer = torch.optim.Adam(vfl_model.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# train
train(vfl_model, EPOCH)
