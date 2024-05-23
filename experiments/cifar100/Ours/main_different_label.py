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
from Cifar100_two import CIFAR100

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

'''
##### Params setting #####
Local: True False False
Vanilla VFL: False True False
Ours: True True True
Ablation with classifier: True True False
Ablation with MAE: False True True
'''
# Train classifier
TRAIN_CLASSIFIER = True
# Train server
TRAIN_SERVER = True
# align loss
align_loss = False
EPOCH = 200
BATCH_SIZE = 500
# learning rate
LR = 0.001
DOWNLOAD_MNIST = True
# 0.1 is the best
LAMBDA = 0.06
# aligned samples
aligned_samples = 8000
# seed
seed = 1

currentDateAndTime = datetime.now()

print("The current date and time is", currentDateAndTime)
path = '/data/data_sw/logs/VFL_base_log/cifar100/Ours/checkpoints/best_cifar100_different_label_' + str(currentDateAndTime) + '_' + str(LAMBDA) + '_align_' + str(align_loss) + '_label_' + str(TRAIN_CLASSIFIER) + '_seed_' + str(seed)
if not os.path.isdir(path):
    os.mkdir(path)
path_A = path + '/client_A.ckpt'
path_B = path + '/client_B.ckpt'
path_server = path + '/server.ckpt'

# log config
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.DEBUG)
logger.setLevel(level=logging.DEBUG)

handler = logging.FileHandler('/data/data_sw/logs/VFL_base_log/cifar100/Ours/logs/' +
                              str(currentDateAndTime) + '_different_label_' + str(aligned_samples) + '_align_' + str(align_loss) + '_label_' + str(TRAIN_CLASSIFIER) + '_seed_' + str(seed) + '.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def setup_seed(seeds):
    torch.manual_seed(seeds)
    torch.cuda.manual_seed_all(seeds)
    np.random.seed(seeds)
    torch.backends.cudnn.deterministic = True


setup_seed(1)

# augmentation
logger.info('==> Preparing data..')
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
# data process
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar100_mean, cifar100_std)
])
train_data = CIFAR100(
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
train_data_aligned, train_data_AB = Data.random_split(train_data, [aligned_samples, len(train_data) - aligned_samples])
train_data_A, train_data_B = Data.random_split(train_data_AB,
                                               [int((len(train_data) - aligned_samples) / 2), int((len(train_data) - aligned_samples) / 2)])

train_data_A_with_aligned = Data.ConcatDataset([train_data_A, train_data_aligned])

train_data_B_with_aligned = Data.ConcatDataset([train_data_B, train_data_aligned])

# test dataset
test_data = CIFAR100(
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
        self.classifier = nn.Linear(512, 20)

    def forward(self, x):
        result = self.classifier(x)
        return result


class VFL_Client(nn.Module):
    def __init__(self):
        super(VFL_Client, self).__init__()
        # padding = (kernel_size-1)/2
        # (1,28,16)
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
        self.Linear = nn.Linear(128 * 4 * 2, 256)
        self.classifier_coarse = nn.Linear(256, 20)
        self.classifier_fine = nn.Linear(256, 100)

    # forward
    def forward(self, x):
        x = self.client_conv(x)
        x = self.Linear(x.view(x.size(0), -1))
        result_coarse = self.classifier_coarse(x)
        result_fine = self.classifier_fine(x)
        return x, result_coarse, result_fine


# Train
def train(client_A, client_B, server, epoch):
    client_A.train()
    client_B.train()
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
    # params for server
    train_loss_server = 0
    correct_server = 0
    total_server = 0

    best_acc_AB = 0
    best_acc_server = 0

    for ep in range(epoch):
        server.train()
        client_A.train()
        client_B.train()
        if TRAIN_CLASSIFIER:
            # train client A
            logger.info('---------------------Train Client A---------------------')
            for batch_idx, (inputs, targets, _) in enumerate(train_A_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                # init grad as 0
                optimizer_A.zero_grad()
                x_client_A, result_A, _ = client_A(inputs[:, :, :, :16])
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
                                                                                            loss.item(), train_acc_A))
            # train client B
            logger.info('---------------------Train Client B---------------------')
            for batch_idx, (inputs, _, targets) in enumerate(train_B_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                # init grad as 0
                optimizer_B.zero_grad()
                x_client_B, _, result_B = client_B(inputs[:, :, :, 16:])
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
                                                                                            loss.item(), train_acc_B))

            # best_acc_server, best_acc_A, best_acc_B = val(client_A, client_B, server, ep, best_acc_A, best_acc_B,
            #                                               best_acc_server, classifier=True)

        # if (ep + 1) % 10 == 0:
        #     best_model_A, best_model_B = nn.DataParallel(VFL_Client()).cuda(), nn.DataParallel(VFL_Client()).cuda()
        #     best_model_A.load_state_dict(torch.load(path_A)['net'])
        #     best_model_B.load_state_dict(torch.load(path_B)['net'])
        #     test(best_model_A, best_model_B, server, ep, classifier=True)

        # train server
        if TRAIN_SERVER:
            logger.info('---------------------Train Server---------------------')
            dataloader_iterator_A = iter(train_A_loader)
            dataloader_iterator_B = iter(train_B_loader)
            for batch_idx, (inputs, targets, _) in enumerate(train_aligned_loader):
                try:
                    (inputs_A, _, _) = next(dataloader_iterator_A)
                except StopIteration:
                    dataloader_iterator_A = iter(train_A_loader)
                    (inputs_A, _, _) = next(dataloader_iterator_A)
                try:
                    (inputs_B, _, _) = next(dataloader_iterator_B)
                except StopIteration:
                    dataloader_iterator_B = iter(train_B_loader)
                    (inputs_B, _, _) = next(dataloader_iterator_B)
                inputs, targets = inputs.cuda(), targets.cuda()
                # init grad as 0
                optimizer_A.zero_grad()
                optimizer_B.zero_grad()
                optimizer_server.zero_grad()
                x_client_A_aligned, result_A, _ = client_A(inputs[:, :, :, :16])
                x_client_B_aligned, result_B, _ = client_B(inputs[:, :, :, 16:])
                result = server(torch.cat([x_client_A_aligned, x_client_B_aligned], dim=1))
                # calculate loss
                loss = loss_func(result, targets)

                # loss backward
                loss.backward()
                # update params
                optimizer_A.step()
                optimizer_B.step()
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

        best_acc_server, best_acc_A, best_acc_B = val(client_A, client_B, server, ep, best_acc_A, best_acc_B,
                                                      best_acc_server)

        if (ep + 1) % 10 == 0:
            best_model_A, best_model_B, best_model_server = nn.DataParallel(VFL_Client()).cuda(), nn.DataParallel(
                VFL_Client()).cuda(), nn.DataParallel(VFL_Server()).cuda(),
            best_model_A.load_state_dict(torch.load(path_A)['net'])
            best_model_B.load_state_dict(torch.load(path_B)['net'])
            best_model_server.load_state_dict(torch.load(path_server)['net'])
            test(best_model_A, best_model_B, best_model_server, ep)

        scheduler_A.step()
        scheduler_B.step()
        scheduler_server.step()


def val(client_A, client_B, server, epoch, best_acc_A=0, best_acc_B=0, best_acc_server=0, classifier=False):
    client_A.eval()
    client_B.eval()
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
    # params for server
    val_loss_server = 0
    correct_server = 0
    total_server = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets_20, targets_100) in enumerate(val_loader):
            inputs, targets_20, targets_100 = inputs.cuda(), targets_20.cuda(), targets_100.cuda()
            x_client_A, result_A, _ = client_A(inputs[:, :, :, :16])
            x_client_B, _, result_B = client_B(inputs[:, :, :, 16:])
            result = server(torch.cat([x_client_A, x_client_B], dim=1))
            loss_A = loss_func(result_A, targets_20)
            loss_B = loss_func(result_B, targets_100)
            loss_server = loss_func(result, targets_20)

            val_loss_A += loss_A.item()
            _, predicted = result_A.max(1)
            total_A += targets_20.size(0)
            correct_A += predicted.eq(targets_20).sum().item()

            val_loss_B += loss_B.item()
            _, predicted = result_B.max(1)
            total_B += targets_100.size(0)
            correct_B += predicted.eq(targets_100).sum().item()

            val_loss_server += loss_server.item()
            _, predicted = result.max(1)
            total_server += targets_20.size(0)
            correct_server += predicted.eq(targets_20).sum().item()

        val_acc_A = correct_A / total_A
        val_acc_B = correct_B / total_B
        val_acc_server = correct_server / total_server

        if classifier:
            logger.info(
                'Epoch: {}, val accuracy_A: {:.4f}, val accuracy_B: {:.4f}'.format(epoch + 1, val_acc_A, val_acc_B))
        else:
            logger.info(
                'Epoch: {}, val accuracy_A: {:.4f}, val accuracy_B: {:.4f}, val accuracy_server: {:.4f}'.format(
                    epoch + 1, val_acc_A, val_acc_B, val_acc_server))

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
        return best_acc_server, best_acc_A, best_acc_B
    else:
        if val_acc_server > best_acc_server:
            logger.info('Save classifiers A...')
            state = {
                'net': client_A.state_dict(),
                'acc': val_acc_A,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, path_A)
            client_A.load_state_dict(torch.load(path_A)['net'])

            logger.info('Save classifiers B...')
            state = {
                'net': client_B.state_dict(),
                'acc': val_acc_B,
                'epoch': epoch,
            }
            torch.save(state, path_B)
            client_B.load_state_dict(torch.load(path_B)['net'])

            logger.info('Save best server..')
            state = {
                'net': server.state_dict(),
                'acc': val_acc_server,
                'epoch': epoch,
            }
            torch.save(state, path_server)
            server.load_state_dict(torch.load(path_server)['net'])
            best_acc_server = val_acc_server
        return best_acc_server, best_acc_A, best_acc_B


def test(client_A, client_B, server, epoch, classifier=False):
    client_A.eval()
    client_B.eval()
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
    # params for server
    test_loss_server = 0
    correct_server = 0
    total_server = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets_20, targets_100) in enumerate(test_loader):
            inputs, targets_20, targets_100 = inputs.cuda(), targets_20.cuda(), targets_100.cuda()
            x_client_A, result_A, _ = client_A(inputs[:, :, :, :16])
            x_client_B, _, result_B = client_B(inputs[:, :, :, 16:])
            result = server(torch.cat([x_client_A, x_client_B], dim=1))
            loss_A = loss_func(result_A, targets_20)
            loss_B = loss_func(result_B, targets_100)
            loss_server = loss_func(result, targets_20)

            test_loss_A += loss_A.item()
            _, predicted = result_A.max(1)
            total_A += targets_100.size(0)
            correct_A += predicted.eq(targets_20).sum().item()

            test_loss_B += loss_B.item()
            _, predicted = result_B.max(1)
            total_B += targets_100.size(0)
            correct_B += predicted.eq(targets_100).sum().item()

            test_loss_server += loss_server.item()
            _, predicted = result.max(1)
            total_server += targets_20.size(0)
            correct_server += predicted.eq(targets_20).sum().item()

        test_acc_A = correct_A / total_A
        test_acc_B = correct_B / total_B
        test_acc_server = correct_server / total_server
        if classifier:
            logger.info(
                'Epoch: {}, test accuracy_A: {:.4f}, test accuracy_B: {:.4f}'.format(epoch + 1, test_acc_A, test_acc_B))
        else:
            logger.info(
                'Epoch: {}, loss_server: {:.4f}, test accuracy_A: {:.4f}, test accuracy_B: {:.4f}, test accuracy_server: {:.4f}'.format(
                    epoch + 1,
                    loss_server,
                    test_acc_A,
                    test_acc_B,
                    test_acc_server))


client_A = nn.DataParallel(VFL_Client()).cuda()
client_B = nn.DataParallel(VFL_Client()).cuda()
server = nn.DataParallel(VFL_Server()).cuda()
optimizer_A = torch.optim.Adam(client_A.parameters(), lr=LR)
optimizer_B = torch.optim.Adam(client_B.parameters(), lr=LR)
optimizer_server = torch.optim.Adam(server.parameters(), lr=LR)

scheduler_A = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_A, T_max=10)
scheduler_B = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_B, T_max=10)
scheduler_server = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_server, T_max=10)

loss_func = nn.CrossEntropyLoss()

# train
train(client_A, client_B, server, EPOCH)
