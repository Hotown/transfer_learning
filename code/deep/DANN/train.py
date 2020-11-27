import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
import tqdm
from torchvision import datasets, transforms

import data_loader
from adv_layer import Discriminator
from model import DANN, Generator

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=.5)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--nepoch', type=int, default=100)
parser.add_argument('--source', type=str, default='mnist')
parser.add_argument('--target', type=str, default='mnist_m')
parser.add_argument('--model_path', type=str, default='models')
parser.add_argument('--result_path', type=str, default='result/result.csv')
args = parser.parse_args()

DEVICE = torch.device('cuda:{}'.format(args.cuda)) if torch.cuda.is_available() else torch.device('cpu')


def test(generator, dataset_name, epoch):
    alpha = 0
    dataloader = data_loader.load_test_data(dataset_name)
    generator.eval()                                                                                                                                                                                                                                        
    n_correct = 0
    with torch.no_grad():
        for _, (t_img, t_label) in enumerate(dataloader):
            t_img, t_label = t_img.to(DEVICE), t_label.to(DEVICE)
            class_output, _ = generator(input_data=t_img)
            prob, pred = torch.max(class_output.data, 1)
            n_correct += (pred == t_label.long()).sum().item()

    acc = float(n_correct) / len(dataloader.dataset) * 100
    return acc

torch.random.manual_seed(10)
loader_src, loader_tar = data_loader.load_data()
# model = DANN(DEVICE).to(DEVICE)
net_G = Generator(DEVICE).to(DEVICE)
net_D = Discriminator(input_dim=50 * 4 * 4, hidden_dim=100).to(DEVICE)
optimizerG = optim.SGD(net_G.parameters(), lr=args.lr)
optimizerD = optim.SGD(net_D.parameters(), lr=args.lr)
loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

best_acc = -float('inf')
len_dataloader = min(len(loader_src), len(loader_tar))
for epoch in range(args.nepoch):
    net_G.train()
    net_D.train()
    i = 1
    for (data_src, data_tar) in tqdm.tqdm(zip(enumerate(loader_src), enumerate(loader_tar)), total=len_dataloader, leave=False):
        _, (x_src, y_src) = data_src
        _, (x_tar, _) = data_tar
        x_src, y_src, x_tar = x_src.to(
            DEVICE), y_src.to(DEVICE), x_tar.to(DEVICE)

        # trian G
        class_output, _ = net_G(input_data=x_src)
        err_s_label = loss_class(class_output, y_src)
        optimizerG.zero_grad()
        err_s_label.backward()
        optimizerG.step()

        # train D
        _, feature_s = net_G(input_data=x_src)
        feature_s = feature_s.to(DEVICE)
        domain_s_output = net_D(feature_s)
        domian_s_label = torch.ones(len(feature_s)).long().to(DEVICE)
        err_s_domain = loss_domain(domain_s_output, domian_s_label)

        _, feature_t = net_G(input_data=x_tar)
        feature_t = feature_t.detach().to(DEVICE)
        domain_t_output = net_D(feature_t)
        domain_t_label = torch.zeros(len(feature_t)).long().to(DEVICE)
        err_t_domain = loss_domain(domain_t_output, domain_t_label)
        err_domain = err_s_domain + err_t_domain
        
        optimizerD.zero_grad()
        err_domain.backward()
        optimizerD.step()
        err = err_s_label + args.gamma * err_domain

        i += 1
    item_pr = 'Epoch: [{}/{}], classify_loss: {:.4f}, domain_loss_s: {:.4f}, domain_loss_t: {:.4f}, domain_loss: {:.4f},total_loss: {:.4f}'.format(
        epoch, args.nepoch, err_s_label.item(), err_s_domain.item(), err_t_domain.item(), err_domain.item(), err.item())
    print(item_pr)
    # fp = open(args.result_path, 'a')
    # fp.write(item_pr + '\n')

    acc_src = test(net_G, args.source, epoch)
    acc_tar = test(net_G, args.target, epoch)
    test_info = 'Source acc: {:.4f}, target acc: {:.4f}'.format(acc_src, acc_tar)
    print(test_info)
    
    if best_acc < acc_tar:
        best_acc = acc_tar
print('Test acc: {:.4f}'.format(best_acc))
