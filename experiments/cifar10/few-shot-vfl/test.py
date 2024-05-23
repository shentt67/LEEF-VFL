from oneshot_vfl import *

path = './checkpoint/best_cifar10.ckpt_2023-07-05 20:51:13.381187_4000'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1)

EPOCH = 70
BATCH_SIZE = 500
# LR = 0.001
LR = 0.001
DOWNLOAD_MNIST = True
# 0.1 is the best
LAMBDA = 0.05

best_model = nn.DataParallel(VFL_Base()).cuda()
print(torch.load(path))
# best_model.load_state_dict(torch.load(path)['net'])
# test(best_model, 0)