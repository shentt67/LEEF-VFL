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
from grad_flip import GradReverse
from tsne_tool import do_tsne

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

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
EPOCH = 10
BATCH_SIZE = 500
# learning rate
LR = 0.001
DOWNLOAD_MNIST = True
# 0.1 is the best
LAMBDA = 1
# aligned samples
aligned_samples = 8000
# seed 1, 10, 100
seed = 1

currentDateAndTime = datetime.now()

'''mmd'''
print("The current date and time is", currentDateAndTime)
# vanilla
# best_cifar10_2023-08-12 13:11:29.694473_500_seed_1
# best_cifar10_2023-08-12 13:11:43.625035_1000_seed_1
# best_cifar10_2023-08-12 13:11:50.568944_2000_seed_1
# best_cifar10_2023-08-12 13:11:55.417470_4000_seed_1
# best_cifar10_2023-08-12 13:11:59.496072_8000_seed_1
# path = '/data/data_sw/logs/VFL_base_log/cifar10/Vanilla/checkpoints/' + 'best_cifar10_2023-08-12 13:11:29.694473_500_seed_1'

# after alignment
# best_cifar10_2023-07-26 13:53:48.025710_0.06_align_True_label_False_seed_1 500
# best_cifar10_2023-07-26 13:53:55.353396_0.06_align_True_label_False_seed_1 1000
# best_cifar10_2023-07-26 13:55:37.133795_0.06_align_True_label_False_seed_1 2000
# best_cifar10_2023-07-26 13:54:46.017350_0.06_align_True_label_False_seed_1 4000
# best_cifar10_2023-09-28 17:08:09.228633_0.02_align_True_label_True_seed_1 8000
path = '/data/data_sw/logs/VFL_base_log/cifar10/Ours/checkpoints/' + 'best_cifar10_2023-09-28 17:08:09.228633_0.02_align_True_label_True_seed_1'

'''tsne'''
# best_cifar10_2023-09-16 14:33:38.124930_0.08_align_True_label_True_seed_1 10ep
# best_cifar10_2023-09-16 14:33:46.895619_0.08_align_True_label_True_seed_1 50ep
# best_cifar10_2023-09-16 14:33:52.683039_0.08_align_True_label_True_seed_1 100ep
# best_cifar10_2023-09-16 14:33:56.224533_0.08_align_True_label_True_seed_1 120ep

'''new tsne'''
# best_cifar10_2023-09-28 20:12:30.839908_0.02_align_True_label_True_seed_1 10ep
# best_cifar10_2023-09-28 20:12:43.846711_0.02_align_True_label_True_seed_1 50ep
# best_cifar10_2023-09-28 20:12:52.186703_0.02_align_True_label_True_seed_1 100ep
# best_cifar10_2023-09-28 15:29:47.987542_0.02_align_True_label_True_seed_1 200ep
# path = '/data/data_sw/logs/VFL_base_log/cifar10/Ours/checkpoints/' + 'best_cifar10_2023-09-28 15:29:47.987542_0.02_align_True_label_True_seed_1'

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

handler = logging.FileHandler('/data/data_sw/logs/VFL_base_log/cifar10/Ours/logs/' +
                              str(currentDateAndTime) + '_' + str(LAMBDA) + '_' + str(
    aligned_samples) + '_align_' + str(align_loss) + '_label_' + str(TRAIN_CLASSIFIER) + '_test.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(seed)

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
train_data_aligned, train_data_AB = Data.random_split(train_data, [aligned_samples, len(train_data) - aligned_samples])
train_data_A, train_data_B = Data.random_split(train_data_AB,
                                               [int((len(train_data) - aligned_samples) / 2),
                                                int((len(train_data) - aligned_samples) / 2)])

train_data_A_with_aligned = Data.ConcatDataset([train_data_A, train_data_aligned])

train_data_B_with_aligned = Data.ConcatDataset([train_data_B, train_data_aligned])

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

# # dataset loader on client A
# train_A_loader = Data.DataLoader(
#     dataset=train_data_A,
#     batch_size=BATCH_SIZE,
#     shuffle=True
# )
# # dataset loader on client B
# train_B_loader = Data.DataLoader(
#     dataset=train_data_B,
#     batch_size=BATCH_SIZE,
#     shuffle=True
# )
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

# tsne loader
tsne_loader = Data.DataLoader(
    dataset=test_data,
    batch_size=10000,
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

        self.Linear = nn.Linear(128 * 4 * 2, 256)
        self.classifier = nn.Linear(256, 10)

    # forward
    def forward(self, x):
        x = self.client_conv(x)
        x = self.Linear(x.view(x.size(0), -1))
        result = self.classifier(x)
        return x, result


def performe_tsne(client_A, client_B, server):
    client_A.eval()
    client_B.eval()
    server.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tsne_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            x_client_A, result_A = client_A(inputs[:, :, :, :16])
            x_client_B, result_B = client_B(inputs[:, :, :, 16:])
            result = server(torch.cat([x_client_A, x_client_B], dim=1))
            # do_tsne(torch.cat([x_client_A, x_client_B], dim=1).detach().cpu(), targets.detach().cpu())
            do_tsne(result.detach().cpu(), targets.detach().cpu())


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
    batch_num = 0

    total_mmd_A = 0
    total_mmd_B = 0
    dataloader_iterator_A = iter(train_A_loader)
    dataloader_iterator_B = iter(train_B_loader)
    with torch.no_grad():
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
            inputs, targets = inputs.cuda(), targets.cuda()
            x_client_A, result_A = client_A(inputs[:, :, :, :16])
            x_client_B, result_B = client_B(inputs[:, :, :, 16:])
            result = server(torch.cat([x_client_A, x_client_B], dim=1))
            loss_A = loss_func(result_A, targets)
            loss_B = loss_func(result_B, targets)
            loss_server = loss_func(result, targets)

            test_loss_A += loss_A.item()
            _, predicted = result_A.max(1)
            total_A += targets.size(0)
            correct_A += predicted.eq(targets).sum().item()

            test_loss_B += loss_B.item()
            _, predicted = result_B.max(1)
            total_B += targets.size(0)
            correct_B += predicted.eq(targets).sum().item()

            test_loss_server += loss_server.item()
            _, predicted = result.max(1)
            total_server += targets.size(0)
            correct_server += predicted.eq(targets).sum().item()

            x_client_A_unaligned, _ = client_A(inputs_A[:, :, :, :16])
            # calculate loss
            mmd_align_A = LAMBDA * mmd(x_client_A, x_client_A_unaligned)
            total_mmd_A += mmd_align_A.sum().item()

            x_client_B_unaligned, _ = client_B(inputs_B[:, :, :, 16:])
            # calculate loss
            mmd_align_B = LAMBDA * mmd(x_client_B, x_client_B_unaligned)
            total_mmd_B += mmd_align_B.sum().item()

            batch_num += 1

        test_acc_A = correct_A / total_A
        test_acc_B = correct_B / total_B
        test_acc_server = correct_server / total_server
        test_mmd_A = total_mmd_A / float(batch_num)
        test_mmd_B = total_mmd_B / float(batch_num)
        if classifier:
            logger.info(
                'Epoch: {}, test accuracy_A: {:.4f}, test accuracy_B: {:.4f}'.format(epoch + 1, test_acc_A, test_acc_B))
        else:
            logger.info(
                'Epoch: {}, test loss: {:.4f}, test accuracy_server: {:.4f}, mmd_A: {:.10f}, mmd_B: {:.10f}'.format(
                    epoch + 1,
                    test_loss_server,
                    test_acc_server,
                    test_mmd_A,
                    test_mmd_B))

best_model_A, best_model_B, best_model_server = nn.DataParallel(VFL_Client()).cuda(), nn.DataParallel(
    VFL_Client()).cuda(), nn.DataParallel(VFL_Server()).cuda(),
best_model_A.load_state_dict(torch.load(path_A)['net'])
best_model_B.load_state_dict(torch.load(path_B)['net'])
best_model_server.load_state_dict(torch.load(path_server)['net'])
loss_func = nn.CrossEntropyLoss()
test(best_model_A, best_model_B, best_model_server, 0)
# performe_tsne(best_model_A, best_model_B, best_model_server)
