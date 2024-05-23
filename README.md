# Build Yourself Before Collaboration: Vertical Federated Learning with Limited Aligned Samples

**The official pytorch implementation of "Build Yourself Before Collaboration: Vertical Federated Learning with Limited Aligned Samples".**

![](./framework.svg)

> [Build Yourself Before Collaboration: Vertical Federated Learning with Limited Aligned Samples]()
> 
> Wei Shen, Mang Ye, Wei Yu, Pong C. Yuen
> 
> Wuhan University, Hong Kong Baptist University
>
> **Abstract** Vertical Federated Learning (VFL) has emerged as a significant privacy-preserving learning paradigm for collaboratively training models with distributed features of shared samples. However, the performance of VFL is hindered when the number of shared/aligned samples is limited. Existing methods attempt to address this challenge with feature generation and pseudo-label estimation for unaligned samples, struggling with unavoidable noise introduced in the generation process. In this work, we propose Local Enhanced Effective Vertical Federated Learning (LEEF-VFL), which fully utilizes the unaligned samples in local client learning before collaboration. Specifically, existing methods deprecate private labels that are irrelevant to the collaboration task. However, we propose using private labels within each client to learn from all the local samples and these private labels, which leads to the construction of robust local models and forms the foundation for effective collaborative learning. Besides, we reveal the distribution bias caused by the limited number of aligned samples and propose mitigating it by minimizing discrepancies in data distribution. The proposed LEEF-VFL is both privacy-preserving and efficient without additional information exchanges and communications. Extensive experiments demonstrate the effectiveness of our method in addressing the adverse influences of limited aligned samples.

## Last Update

**2024/05/23** We have released the official codes.

## Start Guideline

- Clone repo and install requirements.txt with the anaconda environment:

```bash
conda create -n LEEF-VFL python=3.9
conda activate LEEF-VFL
git clone https://github.com/shentt67/LEEF-VFL.git
cd LEEF-VFL
pip install requirements.txt
```

- For instance, to evaluate the performance of LEEF-VFL on CIFAR10, perform:

```bash
cd ./experiments/cifar10/Ours
python main.py
```
