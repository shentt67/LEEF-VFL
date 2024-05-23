import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np
import os
from sklearn.cluster import KMeans

from torchvision.transforms import transforms
import copy
from randaugment import Mask, Noise
import logging
from datetime import datetime
import re

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

EPOCH = 70
BATCH_SIZE = 500
# LR = 0.001
LR = 0.001
DOWNLOAD_MNIST = True
# 0.1 is the best
LAMBDA = 0.05
# class threshold
T = 0.85
# aligned samples
aligned_samples = 8000

currentDateAndTime = datetime.now()

print("The current date and time is", currentDateAndTime)
path = '/data0/data_sw/logs/VFL_base_log/fashion_mnist/few-shot-vfl/checkpoints/best_fashion_mnist.ckpt_' + str(
    currentDateAndTime) + '_' + str(aligned_samples)

# log config
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.DEBUG)
logger.setLevel(level=logging.DEBUG)

handler = logging.FileHandler('/data0/data_sw/logs/VFL_base_log/fashion_mnist/few-shot-vfl/logs/' +
                              str(currentDateAndTime) + '_cluster_' + str(aligned_samples) + '_threshold=' + str(
    T) + '.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1)

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
align_loss = True

# augmentation
logger.info('==> Preparing data..')


class TransformFixMatch(object):
    def __init__(self):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            Mask(p=0.3)
        ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            Mask(p=0.3),
            Noise(miu=0, sigma=1)
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor()])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(x), self.normalize(weak), self.normalize(strong)


# data process
transform = transforms.Compose([
    transforms.ToTensor()
])
train_data = torchvision.datasets.FashionMNIST(
    root='/data0/data_sw/',
    train=True,
    transform=TransformFixMatch(),
    download=True
)

train_data, val_data = Data.random_split(train_data, [55000, 5000])

'''
Divide dataset according to aligned samples
'''
# # aligned samples 500
train_data_aligned, train_data_AB = Data.random_split(train_data, [aligned_samples, 55000 - aligned_samples])
train_data_A, train_data_B = Data.random_split(train_data_AB,
                                               [int((55000 - aligned_samples) / 2), int((55000 - aligned_samples) / 2)])

train_data_A_with_aligned = Data.ConcatDataset([train_data_A, train_data_aligned])

train_data_B_with_aligned = Data.ConcatDataset([train_data_B, train_data_aligned])

# test dataset
test_data = torchvision.datasets.FashionMNIST(
    root='/data0/data_sw/',
    train=False,
    transform=transform
)
# aligned dataset
train_aligned_loader = Data.DataLoader(
    dataset=train_data_aligned,
    batch_size=BATCH_SIZE,
    shuffle=False
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


class VFL_Base(nn.Module):
    def __init__(self):
        super(VFL_Base, self).__init__()
        self.client_A = VFL_Client()
        self.client_B = VFL_Client()
        self.server = VFL_Server()

    def forward(self, x):
        x_client_A = x[:, :, :, :14]
        x_client_B = x[:, :, :, 14:]
        x_client_A, result_local_A = self.client_A(x_client_A)
        x_client_B, result_local_B = self.client_B(x_client_B)
        result, result_A, result_B = self.server([x_client_A, x_client_B])
        return result, result_A, result_B, x_client_A, x_client_B, result_local_A, result_local_B


class VFL_Server(nn.Module):
    def __init__(self):
        super(VFL_Server, self).__init__()
        self.classifier_AB = nn.Linear(512, 10)
        self.classifier_A = nn.Linear(256, 10)
        self.classifier_B = nn.Linear(256, 10)

    def forward(self, x):
        # x = self.aggregation(x)
        result = self.classifier_AB(torch.cat([x[0], x[1]], dim=1))
        result_A = self.classifier_A(x[0])
        result_B = self.classifier_B(x[1])

        return result, result_A, result_B


class VFL_Client(nn.Module):
    def __init__(self):
        super(VFL_Client, self).__init__()
        # padding = (kernel_size-1)/2
        # (1,28,14)
        self.client_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),  # (32,28,14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (32,14,7)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # (64,14,7)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # (64,14,7)
            nn.ReLU()
        )
        self.Linear = nn.Linear(64 * 14 * 7, 256)
        self.classifier = nn.Linear(256, 10)

    # forward
    def forward(self, x):
        x = self.client_conv(x)
        x = self.Linear(x.view(x.size(0), -1))
        result = self.classifier(x)
        return x, result


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

    gradients = []
    # first step: calculate gradient and get pseudo labels
    logger.info('Calculate gradient and get pseudo labels...')
    for batch_idx, ((inputs, _, _), targets) in enumerate(train_aligned_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs.requires_grad_()
        inputs.retain_grad()
        optimizer.zero_grad()
        result, result_A, result_B, x_client_A_aligned, x_client_B_aligned, _, _ = model(inputs)
        # calculate loss
        loss = loss_func(result, targets)
        # loss backward
        loss.backward()
        # store gradient
        gradients.append(inputs.grad)

    gradients = torch.cat(gradients).cpu().numpy()

    # calculate labels by cluster
    num_clusters = 10
    logger.info("Start cluster...")
    gradients = gradients.reshape((gradients.shape[0], gradients.shape[1] * gradients.shape[2] * gradients.shape[3]))
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=1).fit(gradients)
    cluster_labels = kmeans.labels_
    logger.info("Finish cluster.")

    # labeled loader
    local_train_labeled = copy.deepcopy(train_data_aligned)
    local_train_labeled.target = cluster_labels
    local_train_labeled_loader = Data.DataLoader(
        local_train_labeled, batch_size=BATCH_SIZE, shuffle=True
    )
    # unlabeled loader
    train_A_unlabeled_loader = Data.DataLoader(
        dataset=train_data_A,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    train_B_unlabeled_loader = Data.DataLoader(
        dataset=train_data_B,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    for param in model.named_parameters():
        if re.match('module.server', param[0]):
            param[1].requires_grad = False
        elif re.match('module.client_A', param[0]):
            param[1].requires_grad = True
        elif re.match('module.client_B', param[0]):
            param[1].requires_grad = True
    for ep in range(20):
        if TRAIN_CLASSIFIER:
            # train client A
            logger.info('---------------------Train Client A---------------------')
            labeled_iter = iter(local_train_labeled_loader)
            unlabeled_iter = iter(train_A_unlabeled_loader)
            for batch_idx in range(50):
                try:
                    (inputs_x, _, _), targets_x = next(labeled_iter)
                except:
                    labeled_iter = iter(local_train_labeled_loader)
                    (inputs_x, _, _), targets_x = next(labeled_iter)

                try:
                    (_, inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
                except:
                    unlabeled_iter = iter(train_A_unlabeled_loader)
                    (_, inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
                inputs_x, targets_x, inputs_u_w, inputs_u_s = inputs_x.cuda(), targets_x.cuda(), inputs_u_w.cuda(), inputs_u_s.cuda()
                # init grad as 0
                optimizer.zero_grad()
                # calculate label loss
                result_x, _, _, _, _, result_A_x, _ = model(inputs_x)
                Lx = loss_func(result_A_x, targets_x)
                # calculate unlabeled loss
                result_u_w, _, _, _, _, result_A_u_w, _ = model(inputs_u_w)
                result_u_s, _, _, _, _, result_A_u_s, _ = model(inputs_u_s)
                pseudo_label = torch.softmax(result_A_u_w.detach(), dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(T).float()
                Lu = (torch.nn.functional.cross_entropy(result_A_u_s, targets_u,
                                                        reduction='none') * mask).mean()
                loss = Lx + Lu
                # loss backward
                loss.backward()
                # update params
                optimizer.step()
                _, predicted = result_A_x.max(1)
                total_A += targets_x.size(0)
                correct_A += predicted.eq(targets_x).sum().item()

                # train_acc
                train_acc_A = correct_A / total_A
                if batch_idx % 10 == 0:
                    logger.info(
                        'Epoch: {}, {}\{}: train loss_A: {:.4f}, accuracy_A: {:.4f}'.format(ep + 1, batch_idx + 1,
                                                                                            50,
                                                                                            loss.item(), train_acc_A))
            # train client B
            logger.info('---------------------Train Client B---------------------')
            labeled_iter = iter(local_train_labeled_loader)
            unlabeled_iter = iter(train_B_unlabeled_loader)
            for batch_idx in range(50):
                try:
                    (inputs_x, _, _), targets_x = next(labeled_iter)
                except:
                    labeled_iter = iter(local_train_labeled_loader)
                    (inputs_x, _, _), targets_x = next(labeled_iter)

                try:
                    (_, inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
                except:
                    unlabeled_iter = iter(train_B_unlabeled_loader)
                    (_, inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
                inputs_x, targets_x, inputs_u_w, inputs_u_s = inputs_x.cuda(), targets_x.cuda(), inputs_u_w.cuda(), inputs_u_s.cuda()
                # init grad as 0
                optimizer.zero_grad()
                # calculate label loss
                result_x, _, _, _, _, _, result_B_x = model(inputs_x)
                Lx = loss_func(result_B_x, targets_x)
                # calculate unlabeled loss
                result_u_w, _, _, _, _, _, result_B_u_w = model(inputs_u_w)
                result_u_s, _, _, _, _, _, result_B_u_s = model(inputs_u_s)
                pseudo_label = torch.softmax(result_B_u_w.detach(), dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(T).float()
                Lu = (torch.nn.functional.cross_entropy(result_B_u_s, targets_u,
                                                        reduction='none') * mask).mean()
                loss = Lx + Lu
                # loss backward
                loss.backward()
                # update params
                optimizer.step()
                _, predicted = result_B_x.max(1)
                total_B += targets_x.size(0)
                correct_B += predicted.eq(targets_x).sum().item()

                # train_acc
                train_acc_B = correct_B / total_B
                if batch_idx % 10 == 0:
                    logger.info(
                        'Epoch: {}, {}\{}: train loss_B: {:.4f}, accuracy_B: {:.4f}'.format(ep + 1,
                                                                                            batch_idx + 1,
                                                                                            50,
                                                                                            loss.item(),
                                                                                            train_acc_B))
        # save model
        best_acc_server, best_acc_AB = val(model, ep, best_acc_AB, best_acc_server, local=True)
        model.train()
        if (ep + 1) % 10 == 0:
            best_model = nn.DataParallel(VFL_Base()).cuda()
            best_model.load_state_dict(torch.load(path)['net'])
            test(best_model, ep, local=True)

    logger.info('---------------------Train Server---------------------')
    for param in model.named_parameters():
        if re.match('module.server', param[0]):
            param[1].requires_grad = True
        elif re.match('module.client_A', param[0]):
            param[1].requires_grad = False
        elif re.match('module.client_B', param[0]):
            param[1].requires_grad = False
    for ep in range(20):
        # train server
        if TRAIN_SERVER:
            for batch_idx, ((inputs, _, _), targets) in enumerate(train_aligned_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                # init grad as 0
                optimizer.zero_grad()
                result, result_A, result_B, x_client_A_aligned, x_client_B_aligned, _, _ = model(inputs)
                # calculate loss
                loss = loss_func(result, targets)
                loss += loss_func(result_A, targets)
                loss += loss_func(result_B, targets)
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
                    logger.info(
                        'Epoch: {}, {}\{}: train loss_server: {:.4f}, accuracy_server: {:.4f}'.format(ep + 1,
                                                                                                      batch_idx + 1,
                                                                                                      len(train_aligned_loader),
                                                                                                      loss.item(),
                                                                                                      train_acc_server))

        best_acc_server, best_acc_AB = val(model, ep, best_acc_AB, best_acc_server)
        model.train()
        if (ep + 1) % 10 == 0:
            best_model = nn.DataParallel(VFL_Base()).cuda()
            best_model.load_state_dict(torch.load(path)['net'])
            test(best_model, ep)

    logger.info('\nLocal training with pseudo labels...')
    for param in model.named_parameters():
        if re.match('module.server', param[0]):
            param[1].requires_grad = False
        elif re.match('module.client_A', param[0]):
            param[1].requires_grad = True
        elif re.match('module.client_B', param[0]):
            param[1].requires_grad = True
    for ep in range(20):
        if TRAIN_CLASSIFIER:
            # train client A
            logger.info('---------------------Train Client A---------------------')
            labeled_iter = iter(local_train_labeled_loader)
            unlabeled_iter = iter(train_A_unlabeled_loader)
            unlabeled_normal_iter = iter(train_A_unlabeled_loader)
            for batch_idx in range(50):
                try:
                    (inputs_x, _, _), targets_x = next(labeled_iter)
                except:
                    labeled_iter = iter(local_train_labeled_loader)
                    (inputs_x, _, _), targets_x = next(labeled_iter)

                try:
                    (_, inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
                except:
                    unlabeled_iter = iter(train_A_unlabeled_loader)
                    (_, inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

                try:
                    (inputs_u_normal, _, _), _ = next(unlabeled_normal_iter)
                except:
                    unlabeled_normal_iter = iter(train_A_unlabeled_loader)
                    (inputs_u_normal, _, _), _ = next(unlabeled_normal_iter)

                inputs_x, targets_x, inputs_u_w, inputs_u_s = inputs_x.cuda(), targets_x.cuda(), inputs_u_w.cuda(), inputs_u_s.cuda()
                # init grad as 0
                optimizer.zero_grad()
                # calculate label loss
                _, _, _, temp_A_al, temp_B_al, result_A_x, _ = model(inputs_x)
                Lx = loss_func(result_A_x, targets_x)
                # calculate unlabeled loss
                _, _, _, _, _, result_A_u_w, _ = model(inputs_u_w)
                _, _, _, _, _, result_A_u_s, _ = model(inputs_u_s)
                pseudo_label = torch.softmax(result_A_u_w.detach(), dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(T).float()

                # calculate pseudo label loss
                result_u_normal, result_A_u_normal, _, temp_A, _, _, _ = model(inputs_u_normal)
                # calculate temp_B
                temp_B = torch.nn.functional.softmax(
                    temp_A @ temp_A_al.T / np.square(temp_A.shape[1]),
                    dim=1) @ temp_B_al
                # calculate result A, result_AB
                result_A, _, result_AB = model.module.server([temp_A, temp_B])
                max_probs_A, targets_A = torch.max(torch.softmax(result_A.detach(), dim=-1), dim=-1)
                max_probs_AB, targets_AB = torch.max(torch.softmax(result_AB.detach(), dim=-1), dim=-1)
                mask_A = max_probs_A.ge(T).float()
                mask_AB = max_probs_AB.ge(T).float()
                mask_label = mask_A * mask_AB

                L_pseudo = (torch.nn.functional.cross_entropy(result_A, targets_A,
                                                              reduction='none') * mask_label).mean()
                Lu = (torch.nn.functional.cross_entropy(result_A_u_s, targets_u,
                                                        reduction='none') * mask * torch.where(mask_label != 0, 0,
                                                                                               1)).mean()
                loss = Lx + Lu + L_pseudo
                # loss backward
                loss.backward()
                # update params
                optimizer.step()
                _, predicted = result_A_x.max(1)
                total_A += targets_x.size(0)
                correct_A += predicted.eq(targets_x).sum().item()

                # train_acc
                train_acc_A = correct_A / total_A
                if batch_idx % 10 == 0:
                    logger.info(
                        'Epoch: {}, {}\{}: train loss_A: {:.4f}, accuracy_A: {:.4f}'.format(ep + 1, batch_idx + 1,
                                                                                            50,
                                                                                            loss.item(), train_acc_A))
            # train client B
            logger.info('---------------------Train Client B---------------------')
            labeled_iter = iter(local_train_labeled_loader)
            unlabeled_iter = iter(train_B_unlabeled_loader)
            unlabeled_normal_iter = iter(train_B_unlabeled_loader)
            for batch_idx in range(50):
                try:
                    (inputs_x, _, _), targets_x = next(labeled_iter)
                except:
                    labeled_iter = iter(local_train_labeled_loader)
                    (inputs_x, _, _), targets_x = next(labeled_iter)

                try:
                    (_, inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
                except:
                    unlabeled_iter = iter(train_B_unlabeled_loader)
                    (_, inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

                try:
                    (inputs_u_normal, _, _), _ = next(unlabeled_normal_iter)
                except:
                    unlabeled_normal_iter = iter(train_B_unlabeled_loader)
                    (inputs_u_normal, _, _), _ = next(unlabeled_normal_iter)
                inputs_x, targets_x, inputs_u_w, inputs_u_s = inputs_x.cuda(), targets_x.cuda(), inputs_u_w.cuda(), inputs_u_s.cuda()
                # init grad as 0
                optimizer.zero_grad()
                # calculate label loss
                _, _, _, temp_A_al, temp_B_al, _, result_B_x = model(inputs_x)
                Lx = loss_func(result_B_x, targets_x)
                # calculate unlabeled loss
                _, _, _, _, _, _, result_B_u_w = model(inputs_u_w)
                _, _, _, _, _, _, result_B_u_s = model(inputs_u_s)
                pseudo_label = torch.softmax(result_B_u_w.detach(), dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(T).float()

                # calculate pseudo label loss
                result_u_normal, _, result_B_u_normal, _, temp_B, _, _ = model(inputs_u_normal)
                # calculate temp_B
                temp_A = torch.nn.functional.softmax(
                    temp_B @ temp_B_al.T / np.square(temp_B.shape[1]),
                    dim=1) @ temp_A_al
                # calculate result A, result_AB
                _, result_B, result_AB = model.module.server([temp_A, temp_B])
                max_probs_B, targets_B = torch.max(torch.softmax(result_B.detach(), dim=-1), dim=-1)
                max_probs_AB, targets_AB = torch.max(torch.softmax(result_AB.detach(), dim=-1), dim=-1)
                mask_B = max_probs_B.ge(T).float()
                mask_AB = max_probs_AB.ge(T).float()
                mask_label = mask_B * mask_AB

                L_pseudo = (torch.nn.functional.cross_entropy(result_B, targets_B,
                                                              reduction='none') * mask_label).mean()

                Lu = (torch.nn.functional.cross_entropy(result_B_u_s, targets_u,
                                                        reduction='none') * mask * torch.where(mask_label != 0, 0,
                                                                                               1)).mean()
                loss = Lx + Lu + L_pseudo
                # loss backward
                loss.backward()
                # update params
                optimizer.step()
                _, predicted = result_B_x.max(1)
                total_B += targets_x.size(0)
                correct_B += predicted.eq(targets_x).sum().item()

                # train_acc
                train_acc_B = correct_B / total_B
                if batch_idx % 10 == 0:
                    logger.info(
                        'Epoch: {}, {}\{}: train loss_B: {:.4f}, accuracy_B: {:.4f}'.format(ep + 1,
                                                                                            batch_idx + 1,
                                                                                            50,
                                                                                            loss.item(),
                                                                                            train_acc_B))
        # save model
        best_acc_server, best_acc_AB = val(model, ep, best_acc_AB, best_acc_server, local=True)
        model.train()
        if (ep + 1) % 10 == 0:
            best_model = nn.DataParallel(VFL_Base()).cuda()
            best_model.load_state_dict(torch.load(path)['net'])
            test(best_model, ep, local=True)

    logger.info('---------------------Train Server---------------------')
    for param in model.named_parameters():
        if re.match('module.server', param[0]):
            param[1].requires_grad = True
        elif re.match('module.client_A', param[0]):
            param[1].requires_grad = False
        elif re.match('module.client_B', param[0]):
            param[1].requires_grad = False
    for ep in range(20):
        # train server
        if TRAIN_SERVER:
            for batch_idx, ((inputs, _, _), targets) in enumerate(train_aligned_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                # init grad as 0
                optimizer.zero_grad()
                result, result_A, result_B, x_client_A_aligned, x_client_B_aligned, _, _ = model(inputs)
                # calculate loss
                loss = loss_func(result, targets)
                loss += loss_func(result_A, targets)
                loss += loss_func(result_B, targets)
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
                    logger.info(
                        'Epoch: {}, {}\{}: train loss_server: {:.4f}, accuracy_server: {:.4f}'.format(ep + 1,
                                                                                                      batch_idx + 1,
                                                                                                      len(train_aligned_loader),
                                                                                                      loss.item(),
                                                                                                      train_acc_server))

        best_acc_server, best_acc_AB = val(model, ep, best_acc_AB, best_acc_server)
        model.train()
        if (ep + 1) % 10 == 0:
            best_model = nn.DataParallel(VFL_Base()).cuda()
            best_model.load_state_dict(torch.load(path)['net'])
            test(best_model, ep)


def val(model, epoch, best_acc_AB=0, best_acc_server=0, local=False):
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
        for batch_idx, ((inputs, _, _), targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            result, _, _, _, _, result_A, result_B = model(inputs)
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

        if local:
            logger.info(
                'Epoch: {}, val accuracy_A: {:.4f}, val accuracy_B: {:.4f}'.format(epoch + 1, val_acc_A, val_acc_B))
        else:
            logger.info(
                'Epoch: {}, val accuracy_server: {:.4f}'.format(
                    epoch + 1,
                    val_acc_server))

    model.train()
    if local:
        acc_AB = 100. * (val_acc_A + val_acc_B) / 2
        if acc_AB > best_acc_AB:
            logger.info('Save best classifiers..')
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
        acc_server = 100. * val_acc_server
        if acc_server > best_acc_server:
            logger.info('Save best server..')
            state = {
                'net': model.state_dict(),
                'acc': acc_server,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, path)
            model.load_state_dict(torch.load(path)['net'])
            best_acc_server = acc_server
        return best_acc_server, best_acc_AB


def test(model, epoch, local=False):
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
            result, _, _, _, _, result_A, result_B = model(inputs)
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
        if local:
            logger.info(
                'Epoch: {}, test accuracy_A: {:.4f}, test accuracy_B: {:.4f}'.format(epoch + 1, test_acc_A, test_acc_B))
        else:
            logger.info(
                'Epoch: {}, test accuracy_server: {:.4f}'.format(
                    epoch + 1,
                    test_acc_server))

    model.train()


vfl_model = nn.DataParallel(VFL_Base()).cuda()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, vfl_model.parameters()), lr=LR)
loss_func = nn.CrossEntropyLoss()

# train
train(vfl_model, EPOCH)
