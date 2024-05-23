import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np
import os
from torchvision.transforms import transforms
from mmd import *
import logging
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
# align loss
align_loss = True
EPOCH = 200
BATCH_SIZE = 500
# learning rate
LR = 0.001
DOWNLOAD_MNIST = True
# 0.1 is the best
LAMBDA = 0.06
# aligned samples
aligned_samples = 2000

currentDateAndTime = datetime.now()

print("The current date and time is", currentDateAndTime)
path = '/data/data_sw/logs/VFL_base_log/cifar10/Ours/checkpoints/best_cifar10_' + str(currentDateAndTime) + '_' + str(
    LAMBDA) + '_' + 'four_clients'
if not os.path.isdir(path):
    os.mkdir(path)
path_A = path + '/client_A.ckpt'
path_B = path + '/client_B.ckpt'
path_C = path + '/client_C.ckpt'
path_D = path + '/client_D.ckpt'
path_server = path + '/server.ckpt'

# log config
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.DEBUG)
logger.setLevel(level=logging.DEBUG)

handler = logging.FileHandler('/data/data_sw/logs/VFL_base_log/cifar10/Ours/logs/' +
                              str(currentDateAndTime) + '_' + str(aligned_samples) + '_four_clients_' + str(
    LAMBDA) + '_align_' + str(align_loss) + '_label_' + str(TRAIN_CLASSIFIER) + '.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1)

# augmentation
logger.info('==> Preparing data..')
# data process
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])
train_data = torchvision.datasets.CIFAR10(
    root='/data/data_sw/',
    train=True,
    transform=transform,
    download=True
)

train_data, val_data = Data.random_split(train_data, [45000, 5000])

'''
Divide dataset according to aligned samples
'''
# # aligned samples 500
train_data_aligned, train_data_AB = Data.random_split(train_data, [aligned_samples, 45000 - aligned_samples])
train_data_A, train_data_B, train_data_C, train_data_D = Data.random_split(train_data_AB,
                                                                           [int((45000 - aligned_samples) / 4),
                                                                            int((45000 - aligned_samples) / 4),
                                                                            int((45000 - aligned_samples) / 4),
                                                                            int((45000 - aligned_samples) / 4)])

train_data_A_with_aligned = Data.ConcatDataset([train_data_A, train_data_aligned])

train_data_B_with_aligned = Data.ConcatDataset([train_data_B, train_data_aligned])

train_data_C_with_aligned = Data.ConcatDataset([train_data_C, train_data_aligned])

train_data_D_with_aligned = Data.ConcatDataset([train_data_D, train_data_aligned])

# test dataset
test_data = torchvision.datasets.CIFAR10(
    root='/data/data_sw/',
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
# dataset loader on client C
train_C_loader = Data.DataLoader(
    dataset=train_data_C_with_aligned,
    batch_size=BATCH_SIZE,
    shuffle=True
)
# dataset loader on client D
train_D_loader = Data.DataLoader(
    dataset=train_data_D_with_aligned,
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


class VFL_Server(nn.Module):
    def __init__(self):
        super(VFL_Server, self).__init__()
        # self.aggregation = nn.Linear(256, 128)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        # x = self.aggregation(x)
        result = self.classifier(x)
        return result


class VFL_Client(nn.Module):
    def __init__(self):
        super(VFL_Client, self).__init__()
        # padding = (kernel_size-1)/2
        # (3,16,16)
        self.client_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),  # (32,16,16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (32,8,8)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # (64,8,8)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # (64,8,8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (64,4,4)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # (128,4,4)
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # (128,4,4)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (128,2,2)
        )

        self.Linear = nn.Linear(128 * 2 * 2, 128)
        self.classifier = nn.Linear(128, 10)

    # forward
    def forward(self, x):
        x = self.client_conv(x)
        x = self.Linear(x.view(x.size(0), -1))
        result = self.classifier(x)
        return x, result


# Train
def train(client_A, client_B, client_C, client_D, server, epoch):
    client_A.train()
    client_B.train()
    client_C.train()
    client_D.train()
    server.train()
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
    # params for C
    train_loss_C = 0
    correct_C = 0
    total_C = 0
    best_acc_C = 0
    # params for D
    train_loss_D = 0
    correct_D = 0
    total_D = 0
    best_acc_D = 0
    # params for server
    train_loss_server = 0
    correct_server = 0
    total_server = 0

    best_acc_AB = 0
    best_acc_server = 0

    for ep in range(EPOCH):
        client_A.train()
        client_B.train()
        client_C.train()
        client_D.train()
        if TRAIN_CLASSIFIER:
            # train client A
            logger.info('---------------------Train Client A---------------------')
            for batch_idx, (inputs, targets) in enumerate(train_A_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                # init grad as 0
                optimizer_A.zero_grad()
                x_client_A, result_A = client_A(inputs[:, :, :16, :16])
                # calculate loss
                loss = loss_func(result_A, targets)
                # loss backward
                loss.backward()
                # update params
                optimizer_A.step()
                # calculate loss sum
                train_loss_A += loss.item()
                _, predicted = result_A.max(1)
                total_A += targets.size(0)
                correct_A += predicted.eq(targets).sum().item()

                # train_acc
                train_acc_A = correct_A / total_A

                if batch_idx % 20 == 0:
                    logger.info(
                        'Epoch: {}, {}\{}: train loss_A: {:.4f}, accuracy_A: {:.4f}'.format(ep + 1, batch_idx + 1,
                                                                                            len(train_A_loader),
                                                                                            loss.item(),
                                                                                            train_acc_A))
            # train client B
            logger.info('---------------------Train Client B---------------------')
            for batch_idx, (inputs, targets) in enumerate(train_B_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                # init grad as 0
                optimizer_B.zero_grad()
                x_client_B, result_B = client_B(inputs[:, :, 16:, 16:])
                # calculate loss
                loss = loss_func(result_B, targets)
                # loss backward
                loss.backward()
                # update params
                optimizer_B.step()
                # calculate loss sum
                train_loss_B += loss.item()
                _, predicted = result_B.max(1)
                total_B += targets.size(0)
                correct_B += predicted.eq(targets).sum().item()

                # train_acc
                train_acc_B = correct_B / total_B

                if batch_idx % 20 == 0:
                    logger.info(
                        'Epoch: {}, {}\{}: train loss_B: {:.4f}, accuracy_B: {:.4f}'.format(ep + 1, batch_idx + 1,
                                                                                            len(train_B_loader),
                                                                                            loss.item(),
                                                                                            train_acc_B))

            # train client C
            logger.info('---------------------Train Client C---------------------')
            for batch_idx, (inputs, targets) in enumerate(train_C_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                # init grad as 0
                optimizer_C.zero_grad()
                x_client_C, result_C = client_C(inputs[:, :, 16:, :16])
                # calculate loss
                loss = loss_func(result_C, targets)
                # loss backward
                loss.backward()
                # update params
                optimizer_C.step()
                # calculate loss sum
                train_loss_C += loss.item()
                _, predicted = result_C.max(1)
                total_C += targets.size(0)
                correct_C += predicted.eq(targets).sum().item()

                # train_acc
                train_acc_C = correct_C / total_C

                if batch_idx % 20 == 0:
                    logger.info(
                        'Epoch: {}, {}\{}: train loss_C: {:.4f}, accuracy_C: {:.4f}'.format(ep + 1,
                                                                                            batch_idx + 1,
                                                                                            len(train_C_loader),
                                                                                            loss.item(),
                                                                                            train_acc_C))

            # train client D
            logger.info('---------------------Train Client D---------------------')
            for batch_idx, (inputs, targets) in enumerate(train_D_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                # init grad as 0
                optimizer_D.zero_grad()
                x_client_D, result_D = client_D(inputs[:, :, 16:, 16:])
                # calculate loss
                loss = loss_func(result_D, targets)
                # loss backward
                loss.backward()
                # update params
                optimizer_D.step()
                # calculate loss sum
                train_loss_D += loss.item()
                _, predicted = result_D.max(1)
                total_D += targets.size(0)
                correct_D += predicted.eq(targets).sum().item()

                # train_acc
                train_acc_D = correct_D / total_D

                if batch_idx % 20 == 0:
                    logger.info(
                        'Epoch: {}, {}\{}: train loss_D: {:.4f}, accuracy_D: {:.4f}'.format(ep + 1,
                                                                                            batch_idx + 1,
                                                                                            len(train_D_loader),
                                                                                            loss.item(),
                                                                                            train_acc_D))

            best_acc_server, best_acc_A, best_acc_B, best_acc_C, best_acc_D = val(client_A, client_B, client_C, client_D, server, ep,
                                                          best_acc_A, best_acc_B,
                                                          best_acc_C, best_acc_D, best_acc_server, classifier=True)

        # for ep in range(50):
        server.train()
        client_A.train()
        client_B.train()
        client_C.train()
        client_D.train()
        # train server
        if TRAIN_SERVER:
            logger.info('---------------------Train Server---------------------')
            dataloader_iterator_A = iter(train_A_loader)
            dataloader_iterator_B = iter(train_B_loader)
            dataloader_iterator_C = iter(train_C_loader)
            dataloader_iterator_D = iter(train_D_loader)
            for batch_idx, (inputs, targets) in enumerate(train_aligned_loader):
                try:
                    (inputs_A, targets_A) = next(dataloader_iterator_A)
                except StopIteration:
                    dataloader_iterator_A = iter(train_A_loader)
                    (inputs_A, targets_A) = next(dataloader_iterator_A)
                try:
                    (inputs_B, targets_B) = next(dataloader_iterator_B)
                except StopIteration:
                    dataloader_iterator_B = iter(train_B_loader)
                    (inputs_B, targets_B) = next(dataloader_iterator_B)
                try:
                    (inputs_C, targets_C) = next(dataloader_iterator_C)
                except StopIteration:
                    dataloader_iterator_C = iter(train_C_loader)
                    (inputs_B, targets_C) = next(dataloader_iterator_C)
                try:
                    (inputs_D, targets_D) = next(dataloader_iterator_D)
                except StopIteration:
                    dataloader_iterator_D = iter(train_D_loader)
                    (inputs_D, targets_D) = next(dataloader_iterator_D)
                inputs, targets = inputs.cuda(), targets.cuda()
                # init grad as 0
                optimizer_A.zero_grad()
                optimizer_B.zero_grad()
                optimizer_C.zero_grad()
                optimizer_D.zero_grad()
                optimizer_server.zero_grad()
                x_client_A_aligned, result_A = client_A(inputs[:, :, :16, :16])
                x_client_B_aligned, result_B = client_B(inputs[:, :, :16, 16:])
                x_client_C_aligned, result_C = client_C(inputs[:, :, 16:, :16])
                x_client_D_aligned, result_D = client_D(inputs[:, :, 16:, 16:])
                result = server(
                    torch.cat([x_client_A_aligned, x_client_B_aligned, x_client_C_aligned, x_client_D_aligned],
                              dim=1))
                # calculate loss
                loss = loss_func(result, targets)

                if align_loss and ep >= 30:
                    # align A
                    x_client_A_unaligned, _ = client_A(inputs_A[:, :, :16, :16])
                    # calculate loss
                    loss_align_A = LAMBDA * mmd(x_client_A_aligned, x_client_A_unaligned)
                    # align B
                    x_client_B_unaligned, _ = client_B(inputs_B[:, :, :16, 16:])
                    # calculate loss
                    loss_align_B = LAMBDA * mmd(x_client_B_aligned, x_client_B_unaligned)
                    # align C
                    x_client_C_unaligned, _ = client_C(inputs_C[:, :, 16:, :16])
                    # calculate loss
                    loss_align_C = LAMBDA * mmd(x_client_C_aligned, x_client_C_unaligned)
                    # align D
                    x_client_D_unaligned, _ = client_D(inputs_D[:, :, 16:, 16:])
                    # calculate loss
                    loss_align_D = LAMBDA * mmd(x_client_D_aligned, x_client_D_unaligned)
                    loss = loss + loss_align_A + loss_align_B + loss_align_C + loss_align_D
                    # loss backward
                    loss.backward()
                    # update params
                    optimizer_A.step()
                    optimizer_B.step()
                    optimizer_C.step()
                    optimizer_D.step()
                    optimizer_server.step()
                else:
                    # loss backward
                    loss.backward()
                    # update params
                    optimizer_A.step()
                    optimizer_B.step()
                    optimizer_C.step()
                    optimizer_D.step()
                    optimizer_server.step()

                # calculate loss sum
                train_loss_server += loss.item()
                _, predicted = result.max(1)
                total_server += targets.size(0)
                correct_server += predicted.eq(targets).sum().item()

                # train_acc
                train_acc_server = correct_server / total_server

                if batch_idx % 20 == 0:
                    logger.info(
                        'Epoch: {}, {}\{}: train loss_server: {:.4f}, accuracy_server: {:.4f}'.format(ep + 1,
                                                                                                      batch_idx + 1,
                                                                                                      len(train_aligned_loader),
                                                                                                      loss.item(),
                                                                                                      train_acc_server))

        best_acc_server, best_acc_A, best_acc_B, best_acc_C, best_acc_D = val(client_A, client_B, client_C,
                                                                              client_D, server, ep, best_acc_A,
                                                                              best_acc_B, best_acc_C, best_acc_D,
                                                                              best_acc_server)

        if (ep + 1) % 10 == 0:
            best_model_A, best_model_B, best_model_C, best_model_D, best_model_server = nn.DataParallel(VFL_Client()).cuda(), nn.DataParallel(
                VFL_Client()).cuda(), nn.DataParallel(VFL_Client()).cuda(), nn.DataParallel(
                VFL_Client()).cuda(), nn.DataParallel(VFL_Server()).cuda(),
            best_model_A.load_state_dict(torch.load(path_A)['net'])
            best_model_B.load_state_dict(torch.load(path_B)['net'])
            best_model_C.load_state_dict(torch.load(path_A)['net'])
            best_model_D.load_state_dict(torch.load(path_B)['net'])
            best_model_server.load_state_dict(torch.load(path_server)['net'])
            test(best_model_A, best_model_B, best_model_C, best_model_D, best_model_server, ep)


def val(client_A, client_B, client_C, client_D, server, epoch, best_acc_A=0, best_acc_B=0, best_acc_C=0, best_acc_D=0,
        best_acc_server=0, classifier=False):
    client_A.eval()
    client_B.eval()
    client_C.eval()
    client_D.eval()
    server.eval()
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
    # params for C
    val_loss_C = 0
    correct_C = 0
    total_C = 0
    # params for D
    val_loss_D = 0
    correct_D = 0
    total_D = 0
    # params for server
    val_loss_server = 0
    correct_server = 0
    total_server = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            x_client_A, result_A = client_A(inputs[:, :, :16, :16])
            x_client_B, result_B = client_B(inputs[:, :, :16, 16:])
            x_client_C, result_C = client_C(inputs[:, :, 16:, :16])
            x_client_D, result_D = client_D(inputs[:, :, 16:, 16:])
            result = server(torch.cat([x_client_A, x_client_B, x_client_C, x_client_D], dim=1))
            loss_A = loss_func(result_A, targets)
            loss_B = loss_func(result_B, targets)
            loss_C = loss_func(result_C, targets)
            loss_D = loss_func(result_D, targets)
            loss_server = loss_func(result, targets)

            val_loss_A += loss_A.item()
            _, predicted = result_A.max(1)
            total_A += targets.size(0)
            correct_A += predicted.eq(targets).sum().item()

            val_loss_B += loss_B.item()
            _, predicted = result_B.max(1)
            total_B += targets.size(0)
            correct_B += predicted.eq(targets).sum().item()

            val_loss_C += loss_C.item()
            _, predicted = result_C.max(1)
            total_C += targets.size(0)
            correct_C += predicted.eq(targets).sum().item()

            val_loss_D += loss_D.item()
            _, predicted = result_D.max(1)
            total_D += targets.size(0)
            correct_D += predicted.eq(targets).sum().item()

            val_loss_server += loss_server.item()
            _, predicted = result.max(1)
            total_server += targets.size(0)
            correct_server += predicted.eq(targets).sum().item()

        val_acc_A = correct_A / total_A
        val_acc_B = correct_B / total_B
        val_acc_C = correct_C / total_C
        val_acc_D = correct_D / total_D
        val_acc_server = correct_server / total_server

        if classifier:
            logger.info(
                'Epoch: {}, val accuracy_A: {:.4f}, val accuracy_B: {:.4f}, val accuracy_C: {:.4f}, val accuracy_D: {:.4f}'.format(
                    epoch + 1, val_acc_A, val_acc_B, val_acc_C, val_acc_D))
        else:
            logger.info(
                'Epoch: {}, val accuracy_server: {:.4f}'.format(
                    epoch + 1,
                    val_acc_server))

    if classifier:
        if val_acc_A > best_acc_A:
            logger.info('Save best classifiers A...')
            state = {
                'net': client_A.state_dict(),
                'acc': val_acc_A,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, path_A)
            client_A.load_state_dict(torch.load(path_A)['net'])
            best_acc_A = val_acc_A

        if val_acc_B > best_acc_B:
            logger.info('Save best classifiers B...')
            state = {
                'net': client_B.state_dict(),
                'acc': val_acc_B,
                'epoch': epoch,
            }
            torch.save(state, path_B)
            client_B.load_state_dict(torch.load(path_B)['net'])
            best_acc_B = val_acc_B

        if val_acc_C > best_acc_C:
            logger.info('Save best classifiers C...')
            state = {
                'net': client_C.state_dict(),
                'acc': val_acc_C,
                'epoch': epoch,
            }
            torch.save(state, path_C)
            client_C.load_state_dict(torch.load(path_C)['net'])
            best_acc_C = val_acc_C

        if val_acc_D > best_acc_D:
            logger.info('Save best classifiers D...')
            state = {
                'net': client_D.state_dict(),
                'acc': val_acc_D,
                'epoch': epoch,
            }
            torch.save(state, path_D)
            client_D.load_state_dict(torch.load(path_D)['net'])
            best_acc_D = val_acc_D
        return best_acc_server, best_acc_A, best_acc_B, best_acc_C, best_acc_D
    else:
        if val_acc_server > best_acc_server:
            logger.info('Save best classifiers A...')
            state = {
                'net': client_A.state_dict(),
                'acc': val_acc_A,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, path_A)
            client_A.load_state_dict(torch.load(path_A)['net'])

            logger.info('Save best classifiers B...')
            state = {
                'net': client_B.state_dict(),
                'acc': val_acc_B,
                'epoch': epoch,
            }
            torch.save(state, path_B)
            client_B.load_state_dict(torch.load(path_B)['net'])

            logger.info('Save best classifiers C...')
            state = {
                'net': client_C.state_dict(),
                'acc': val_acc_C,
                'epoch': epoch,
            }
            torch.save(state, path_C)
            client_C.load_state_dict(torch.load(path_C)['net'])

            logger.info('Save best classifiers D...')
            state = {
                'net': client_D.state_dict(),
                'acc': val_acc_D,
                'epoch': epoch,
            }
            torch.save(state, path_D)
            client_D.load_state_dict(torch.load(path_D)['net'])

            logger.info('Save best server..')
            state = {
                'net': server.state_dict(),
                'acc': val_acc_server,
                'epoch': epoch,
            }
            torch.save(state, path_server)
            server.load_state_dict(torch.load(path_server)['net'])
            best_acc_server = val_acc_server
        return best_acc_server, best_acc_A, best_acc_B, best_acc_C, best_acc_D


def test(client_A, client_B, client_C, client_D, server, epoch, classifier=False):
    client_A.eval()
    client_B.eval()
    client_C.eval()
    client_D.eval()
    server.eval()
    test_loss = 0
    correct = 0
    total = 0
    # params for A
    test_loss_A = 0
    correct_A = 0
    total_A = 0
    # params for B
    test_loss_B = 0
    correct_B = 0
    total_B = 0
    # params for C
    test_loss_C = 0
    correct_C = 0
    total_C = 0
    # params for D
    test_loss_D = 0
    correct_D = 0
    total_D = 0
    # params for server
    test_loss_server = 0
    correct_server = 0
    total_server = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            x_client_A, result_A = client_A(inputs[:, :, :16, :16])
            x_client_B, result_B = client_B(inputs[:, :, :16, 16:])
            x_client_C, result_C = client_C(inputs[:, :, 16:, :16])
            x_client_D, result_D = client_D(inputs[:, :, 16:, 16:])
            result = server(torch.cat([x_client_A, x_client_B, x_client_C, x_client_D], dim=1))
            loss_A = loss_func(result_A, targets)
            loss_B = loss_func(result_B, targets)
            loss_C = loss_func(result_A, targets)
            loss_D = loss_func(result_B, targets)
            loss_server = loss_func(result, targets)

            test_loss_A += loss_A.item()
            _, predicted = result_A.max(1)
            total_A += targets.size(0)
            correct_A += predicted.eq(targets).sum().item()

            test_loss_B += loss_B.item()
            _, predicted = result_B.max(1)
            total_B += targets.size(0)
            correct_B += predicted.eq(targets).sum().item()

            test_loss_C += loss_C.item()
            _, predicted = result_C.max(1)
            total_C += targets.size(0)
            correct_C += predicted.eq(targets).sum().item()

            test_loss_D += loss_D.item()
            _, predicted = result_D.max(1)
            total_D += targets.size(0)
            correct_D += predicted.eq(targets).sum().item()

            test_loss_server += loss_server.item()
            _, predicted = result.max(1)
            total_server += targets.size(0)
            correct_server += predicted.eq(targets).sum().item()

        test_acc_A = correct_A / total_A
        test_acc_B = correct_B / total_B
        test_acc_C = correct_C / total_C
        test_acc_D = correct_D / total_D
        test_acc_server = correct_server / total_server
        if classifier:
            logger.info(
                'Epoch: {}, test accuracy_A: {:.4f}, test accuracy_B: {:.4f}, test accuracy_C: {:.4f}, test accuracy_D: {:.4f}'.format(
                    epoch + 1, test_acc_A, test_acc_B, test_acc_C, test_acc_D))
        else:
            logger.info(
                'Epoch: {}, test accuracy_server: {:.4f}'.format(
                    epoch + 1,
                    test_acc_server))


client_A = nn.DataParallel(VFL_Client()).cuda()
client_B = nn.DataParallel(VFL_Client()).cuda()
client_C = nn.DataParallel(VFL_Client()).cuda()
client_D = nn.DataParallel(VFL_Client()).cuda()
server = nn.DataParallel(VFL_Server()).cuda()
optimizer_A = torch.optim.Adam(client_A.parameters(), lr=LR)
optimizer_B = torch.optim.Adam(client_B.parameters(), lr=LR)
optimizer_C = torch.optim.Adam(client_C.parameters(), lr=LR)
optimizer_D = torch.optim.Adam(client_D.parameters(), lr=LR)
optimizer_server = torch.optim.Adam(server.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# train
train(client_A, client_B, client_C, client_D, server, EPOCH)
