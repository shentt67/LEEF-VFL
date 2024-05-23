import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np
import os
from torchvision.transforms import transforms
from datetime import datetime
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5'
EPOCH = 70
BATCH_SIZE = 500
LR = 0.001
# threshold
T = 0.6
# sharpen temperature
t = 0.1
# aligned samples
aligned_samples = 8000
# lambda
Lambda1, Lambda2, Lambda3, Lambda4, Lambda5 = 0.1, 0.1, 0.1, 0.1, 0.1

currentDateAndTime = datetime.now()

# log config
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.DEBUG)
logger.setLevel(level=logging.DEBUG)
handler = logging.FileHandler('/data0/data_sw/logs/VFL_base_log/mnist/FedCVT/logs/' +
                              str(currentDateAndTime) + '_FedCVT_' + str(aligned_samples) + '_threshold=' + str(T) + '.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

currentDateAndTime = datetime.now()

print("The current date and time is", currentDateAndTime)
path = '/data0/data_sw/logs/VFL_base_log/mnist/FedCVT/checkpoints/best_mnist.ckpt_' + str(currentDateAndTime) + '_' + str(aligned_samples)


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
TRAIN_CLASSIFIER = False
# Train server
TRAIN_SERVER = True

# augmentation
print('==> Preparing data..')
# data process
transform = transforms.Compose([
    transforms.ToTensor()
])
train_data = torchvision.datasets.MNIST(
    root='/data0/data_sw/',
    train=True,
    transform=transform,
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
test_data = torchvision.datasets.MNIST(
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
# unaligned samples in A
train_A_unaligned_loader = Data.DataLoader(
    dataset=train_data_A,
    batch_size=BATCH_SIZE,
    shuffle=True
)
# unaligned samples in B
train_B_unaligned_loader = Data.DataLoader(
    dataset=train_data_B,
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
        x_unique_A, x_common_A = self.client_A(x_client_A)
        x_unique_B, x_common_B = self.client_B(x_client_B)
        inputs = [x_unique_A, x_common_A, x_unique_B, x_common_B]
        result_A, result_B, result_AB = self.server(inputs)
        return result_A, result_B, result_AB, x_unique_A, x_common_A, x_unique_B, x_common_B


class VFL_Server(nn.Module):
    def __init__(self):
        super(VFL_Server, self).__init__()
        self.classifier_AB = nn.Linear(256, 10)
        self.classifier_A = nn.Linear(128, 10)
        self.classifier_B = nn.Linear(128, 10)

    def forward(self, x):
        result_A = self.classifier_A(torch.cat([x[0], x[1]], dim=1))
        result_B = self.classifier_B(torch.cat([x[2], x[3]], dim=1))
        result_AB = self.classifier_AB(torch.cat([x[0], x[1], x[2], x[3]], dim=1))
        return result_A, result_B, result_AB


class VFL_Client(nn.Module):
    def __init__(self):
        super(VFL_Client, self).__init__()
        # padding = (kernel_size-1)/2
        # (3,32,14)
        self.client_conv_unique = nn.Sequential(
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

        self.client_conv_common = nn.Sequential(
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

        self.Linear_unique = nn.Linear(64 * 14 * 7, 64)
        self.Linear_common = nn.Linear(64 * 14 * 7, 64)

    # forward
    def forward(self, x):
        # calculate unique
        x_unique = self.client_conv_unique(x)
        x_unique = self.Linear_unique(x_unique.view(x_unique.size(0), -1))
        # calculate common
        x_common = self.client_conv_common(x)
        x_common = self.Linear_common(x_common.view(x_common.size(0), -1))
        # return two embeddings
        return x_unique, x_common


# Train
def train(model, epoch):
    mse_loss = nn.MSELoss(reduction='mean')
    ce_loss = nn.CrossEntropyLoss()
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

    logger.info("Start Training...")
    for ep in range(epoch):
        aligned_iter = iter(train_aligned_loader)
        A_iter = iter(train_A_unaligned_loader)
        B_iter = iter(train_B_unaligned_loader)

        for batch_idx in range(50):
            try:
                inputs_X, targets_X = next(aligned_iter)
            except:
                aligned_iter = iter(train_aligned_loader)
                inputs_X, targets_X = next(aligned_iter)

            try:
                inputs_A, targets_A = next(A_iter)
            except:
                A_iter = iter(train_A_unaligned_loader)
                inputs_A, targets_A = next(A_iter)

            try:
                inputs_B, targets_B = next(B_iter)
            except:
                B_iter = iter(train_B_unaligned_loader)
                inputs_B, targets_B = next(B_iter)

            inputs_X, targets_X, inputs_A, targets_A, inputs_B, targets_B = \
                inputs_X.cuda(), targets_X.cuda(), inputs_A.cuda(), targets_A.cuda(), inputs_B.cuda(), targets_B.cuda()
            # init grad as 0
            optimizer.zero_grad()
            # calculate loss for aligned samples
            result_A, result_B, result_AB, x_unique_A_al, x_common_A_al, x_unique_B_al, x_common_B_al = model(inputs_X)

            _, predicted = result_AB.max(1)
            total_server += targets_X.size(0)
            correct_server += predicted.eq(targets_X).sum().item()

            # train_acc
            train_acc = correct_server / total_server

            L_c_A = ce_loss(result_A, targets_X)
            L_c_B = ce_loss(result_B, targets_X)
            L_c_AB = ce_loss(result_AB, targets_X)
            L_dif_aligned = mse_loss(x_common_A_al, x_common_B_al)
            L_sim_aligned_A = torch.mean(torch.sum(torch.norm(x_unique_A_al * x_common_A_al, p=2, dim=1), dim=-1))
            L_sim_aligned_B = torch.mean(torch.sum(torch.norm(x_unique_B_al * x_common_B_al, p=2, dim=1), dim=-1))

            # calculate loss for unaligned samples A
            _, _, _, x_unique_A_nl, x_common_A_nl, _, _ = model(inputs_A)
            _, _, _, _, _, x_unique_B_nl, x_common_B_nl = model(inputs_B)
            x_common_A = torch.cat([x_common_A_al, x_common_A_nl], dim=0)
            x_common_B = torch.cat([x_common_B_al, x_common_B_nl], dim=0)
            # for A, calculate common B
            x_common_B_nl_for_A = torch.nn.functional.softmax(
                x_common_A_nl @ x_common_B.T / np.square(x_common_A_nl.shape[1]),
                dim=1) @ x_common_B
            # for A, calculate unique B
            x_unique_B_nl_for_A = torch.nn.functional.softmax(
                x_unique_A_nl @ x_unique_A_al.T / np.square(x_unique_A_nl.shape[1]),
                dim=1) @ x_unique_B_al
            # calculate result_A, result_B, result_AB
            result_A, result_B, result_AB = model.module.server(
                [x_unique_A_nl, x_common_A_nl, x_unique_B_nl_for_A, x_common_B_nl_for_A])
            max_probs_A, targets_A = torch.max(torch.softmax(result_A.detach() / t, dim=-1), dim=-1)
            max_probs_B, targets_B = torch.max(torch.softmax(result_B.detach() / t, dim=-1), dim=-1)
            max_probs_AB, targets_AB = torch.max(torch.softmax(result_AB.detach() / t, dim=-1), dim=-1)
            mask_A = max_probs_A.ge(T).float()
            mask_B = max_probs_B.ge(T).float()
            mask_AB = max_probs_AB.ge(T).float()
            mask_label = mask_A * mask_B * mask_AB

            L_pseudo_A = (torch.nn.functional.cross_entropy(result_AB, targets_AB,
                                                            reduction='none') * mask_label).mean()

            # calculate loss for unaligned samples B
            # for B, calculate common A
            x_common_A_nl_for_B = torch.nn.functional.softmax(
                x_common_B_nl @ x_common_A.T / np.square(x_common_B_nl.shape[1]),
                dim=1) @ x_common_A
            # for B, calculate unique A
            x_unique_A_nl_for_B = torch.nn.functional.softmax(
                x_unique_B_nl @ x_unique_B_al.T / np.square(x_unique_B_nl.shape[1]),
                dim=1) @ x_unique_A_al
            # calculate result_A, result_B, result_AB
            result_A, result_B, result_AB = model.module.server(
                [x_unique_A_nl_for_B, x_common_A_nl_for_B, x_unique_B_nl, x_common_B_nl])
            max_probs_A, targets_A = torch.max(torch.softmax(result_A.detach(), dim=-1), dim=-1)
            max_probs_B, targets_B = torch.max(torch.softmax(result_B.detach(), dim=-1), dim=-1)
            max_probs_AB, targets_AB = torch.max(torch.softmax(result_AB.detach(), dim=-1), dim=-1)
            mask_A = max_probs_A.ge(T).float()
            mask_B = max_probs_B.ge(T).float()
            mask_AB = max_probs_AB.ge(T).float()
            mask_label = mask_A * mask_B * mask_AB

            L_pseudo_B = (torch.nn.functional.cross_entropy(result_AB, targets_AB,
                                                            reduction='none') * mask_label).mean()

            # for A, calculate common B
            x_common_B_al_for_A = torch.nn.functional.softmax(
                x_common_A_al @ x_common_B.T / np.square(x_common_A_al.shape[1]),
                dim=1) @ x_common_B
            # for A, calculate unique B
            x_unique_B_al_for_A = torch.nn.functional.softmax(
                x_unique_A_al @ x_unique_A_al.T / np.square(x_unique_A_al.shape[1]),
                dim=1) @ x_unique_B_al

            # for B, calculate common A
            x_common_A_al_for_B = torch.nn.functional.softmax(
                x_common_B_al @ x_common_A.T / np.square(x_common_B_al.shape[1]),
                dim=1) @ x_common_A
            # for B, calculate unique A
            x_unique_A_al_for_B = torch.nn.functional.softmax(
                x_unique_B_al @ x_unique_B_al.T / np.square(x_unique_B_al.shape[1]),
                dim=1) @ x_unique_A_al

            L_dif_A_estimate = mse_loss(
                torch.cat([x_unique_A_al_for_B, x_common_A_al_for_B], dim=1),
                torch.cat([x_unique_A_al, x_common_A_al], dim=1))
            L_dif_B_estimate = mse_loss(
                torch.cat([x_unique_B_al_for_A, x_common_B_al_for_A], dim=1),
                torch.cat([x_unique_B_al, x_common_B_al], dim=1))

            loss = (
                           L_c_AB + L_pseudo_A + L_pseudo_B) + L_c_A + L_c_B + Lambda1 * L_dif_aligned + Lambda2 * L_dif_A_estimate + Lambda3 * L_dif_B_estimate + Lambda4 * L_sim_aligned_A + Lambda5 * L_sim_aligned_B
            # loss backward
            loss.backward()
            # update params
            optimizer.step()
            if batch_idx % 10 == 0:
                logger.info(
                    'Epoch: {}, {}\{}: train loss: {:.4f}, accuracy: {:.4f}'.format(ep + 1, batch_idx + 1,
                                                                                    50,
                                                                                    loss.item(), train_acc))

        best_acc_server, best_acc_AB = val(model, ep, best_acc_AB, best_acc_server)

        model.train()
        if (ep + 1) % 10 == 0:
            best_model = nn.DataParallel(VFL_Base()).cuda()
            best_model.load_state_dict(torch.load(path)['net'])
            test(best_model, ep)


def val(model, epoch, best_acc_AB=0, best_acc_server=0, classifier=False):
    ce_loss = nn.CrossEntropyLoss()
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
            result_A, result_B, result, _, _, _, _ = model(inputs)
            loss_A = ce_loss(result_A, targets)
            loss_B = ce_loss(result_A, targets)
            loss_server = ce_loss(result_A, targets)

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
        logger.info(
            'Epoch: {}, val accuracy_A: {:.4f}, val accuracy_B: {:.4f}, val accuracy_server: {:.4f}'.format(epoch + 1,
                                                                                                            val_acc_A,
                                                                                                            val_acc_B,
                                                                                                            val_acc_server))
    model.train()
    if classifier:
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
        acc_AB = 100. * 100. * (val_acc_A + val_acc_B) / 2
        acc_server = 100. * val_acc_server
        # if acc_server > best_acc_server and acc_AB > best_acc_AB:
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
            best_acc_AB = acc_AB
            best_acc_server = acc_server
        return best_acc_server, best_acc_AB


def test(model, epoch):
    model.eval()
    ce_loss = nn.CrossEntropyLoss()
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
            result_A, result_B, result, _, _, _, _ = model(inputs)
            loss_A = ce_loss(result_A, targets)
            loss_B = ce_loss(result_B, targets)
            loss = ce_loss(result, targets)

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
        logger.info(
            'Epoch: {}, test accuracy_A: {:.4f}, test accuracy_B: {:.4f}, test accuracy_server: {:.4f}'.format(
                epoch + 1, test_acc_A, test_acc_B, test_acc_server))
    model.train()


vfl_model = nn.DataParallel(VFL_Base()).cuda()
optimizer = torch.optim.Adam(vfl_model.parameters(), lr=LR)

# train
train(vfl_model, EPOCH)
