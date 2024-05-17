import argparse
import os

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils
from datasets import get_dataset
from loss import get_loss
from model import Model


def train(net, data_loader, loss_criterion, train_optimizer, batch_size, *, writer, cuda=True, step=0):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, data_loader
    for pos_1, pos_2, target in train_bar:
        if cuda:
            pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        _, out_1 = net(pos_1)
        _, out_2 = net(pos_2)

        out_m = [out_1, out_2]
        losses = []
        for i in range(len(out_m)):
            for j in range(i + 1, len(out_m)):
                losses.append(loss_criterion(out_m[i], out_m[j], [], target))
        loss = torch.stack(losses).mean()
        
        step += 1
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        global_grad_norm = torch.zeros(1)
        if cuda:
            global_grad_norm = global_grad_norm.cuda()
        for group in train_optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad = p.grad
                    global_grad_norm.add_(grad.pow(2).sum())
        global_grad_norm = torch.sqrt(global_grad_norm).cpu()

        writer.add_scalar("loss/train", loss, step)
        writer.add_scalar("grad/global_grad_norm", global_grad_norm, step)

        total_num += batch_size
        total_loss += loss.item() * batch_size
    return total_loss / total_num, step


def test(net, memory_data_loader, test_data_loader, *, top_k, class_cnt, cuda=True, temperature):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        for data, _, target in memory_data_loader:
            if cuda:
                data = data.cuda(non_blocking=True)
            feature, _ = net(data)
            feature_bank.append(feature)
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        feature_labels = torch.tensor(
            memory_data_loader.dataset.labels, device=feature_bank.device)
        test_bar = test_data_loader
        for data, _, target in test_bar:
            if cuda:
                data, target = data.cuda(
                    non_blocking=True), target.cuda(non_blocking=True)
            feature, _ = net(data)

            total_num += data.size(0)
            sim_matrix = torch.mm(feature, feature_bank)
            sim_weight, sim_indices = sim_matrix.topk(k=top_k, dim=-1)
            sim_labels = torch.gather(feature_labels.expand(
                data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            one_hot_label = torch.zeros(
                data.size(0) * top_k, class_cnt, device=sim_labels.device)
            one_hot_label = one_hot_label.scatter(
                dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, class_cnt) * sim_weight.unsqueeze(dim=-1),
                                    dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum(
                (pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum(
                (pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

    return total_top1 / total_num * 100, total_top5 / total_num * 100


def main(dataset: str, loss: str, root: str, root_out: str, batch_size: int,
         model_arch, *, cuda=True, feature_dim=128, temperature=0.5,
         tau_plus=0.1, top_k=200, epochs=200, lr=1e-3, weight_decay=1e-6, writer):

    train_loader = DataLoader(
        get_dataset(dataset, root=root, split="train+unlabeled",
                    transform=utils.train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    memory_loader = DataLoader(
        get_dataset(
            dataset, root=root, split="train", transform=utils.test_transform),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(
        get_dataset(dataset, root=root, split="test", transform=utils.test_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    loss_criterion = get_loss(loss, temperature, cuda, tau_plus)
    print("Loss: ", loss_criterion)

    # Модель и оптимайзер
    model = Model(feature_dim, model_arch)
    if cuda:
        model = model.cuda()
    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    c = len(memory_loader.dataset.classes)
    print("Classes: ", c)

    step = 0
    log_acc1 = []
    log_acc5 = []
    for epoch in tqdm(range(1, epochs + 1)):
        _, step = train(model, train_loader, loss_criterion, optimizer, batch_size,
                                 cuda=cuda, step=step, writer=writer)
        if epoch % 1 == 0:
            test_acc_1, test_acc_5 = test(model, memory_loader, test_loader, cuda=cuda, class_cnt=c, top_k=top_k,
                                          temperature=temperature)
            writer.add_scalar("loss/acc1", test_acc_1, epoch)
            writer.add_scalar("loss/acc5", test_acc_5, epoch)
            log_acc1.append(test_acc_1)
            log_acc5.append(test_acc_5)

            model_path = "{}/model_{}.pth".format(root_out, epoch)
            torch.save(model.state_dict(), model_path)
            print(f"Epoch {epoch}, acc1: {test_acc_1}")

    # Сохранение логов
    with open('{}/acc1.txt'.format(root_out), "w") as file:
        for acc in log_acc1:
            file.write(str(acc) + "\n")
    with open('{}/acc5.txt'.format(root_out), "w") as file:
        for acc in log_acc5:
            file.write(str(acc) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Base model')
    parser.add_argument('--dataset', type=str, help='Название датасета (CIFAR10)')
    parser.add_argument(
        '--loss', type=str, help='Название лосс-функции (Contrastive или DebiasedPos)')
    parser.add_argument('--input_dir', type=str, help='Расположение датасета')
    parser.add_argument('--output_dir', type=str, help='Путь для сохранения логов и моделей')
    parser.add_argument('--encoder', type=str,
                        help='Энкодер (resnet18/34/50')
    parser.add_argument('--cuda', help='Use cuda', action=argparse.BooleanOptionalAction)
    parser.add_argument('--feature_dim', default=128,
                        type=int, help='Размерность скрытого пространства')
    parser.add_argument('--temperature', default=0.5,
                        type=float, help='Температура в софтмаксе')
    parser.add_argument('--tau_plus', default=0.1,
                        type=float, help='Приоритет положительного класса')
    parser.add_argument('--top_k', default=200, type=int,
                        help='Сколько ближайших соседей используется для предсказания')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Размер батча')
    parser.add_argument('--epochs', default=500, type=int,
                        help='Число эпох')
    parser.add_argument('--lr', default=1e-3, type=float, )
    parser.add_argument('--weight_decay', default=1e-6, type=float,)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    writer = SummaryWriter(log_dir="/content/tensorboard/DebiasedPos_loss")
    main(
        args.dataset, args.loss, args.input_dir, args.output_dir, args.batch_size,
        args.encoder, cuda=args.cuda, feature_dim=args.feature_dim,
        temperature=args.temperature, tau_plus=args.tau_plus, top_k=args.top_k,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, writer=writer
    )
