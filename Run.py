import time
import torch
import pandas as pd
import polars as pl
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.amp import autocast, GradScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_eval_iters(dataloader):
    dataset_size = len(dataloader.dataset)
    if dataset_size > 10000:
        return min(100, len(dataloader))
    else:
        return min(50, len(dataloader))

def _loss(model, dataloader):
    eval_iters = get_eval_iters(dataloader)
    loss = []
    acc = []
    data_iter = iter(dataloader)

    for t in range(eval_iters):
        try:
            inputs, labels = next(data_iter)
        except StopIteration:
            break

        inputs, labels = inputs.to(device), labels.to(device)
        B, C, H, W = inputs.shape
        logits = model(inputs)
        loss.append(F.cross_entropy(logits, labels))
        pred = torch.argmax(logits, dim=-1)
        acc.append((pred == labels).sum() / B)

    if len(loss) == 0:
        return {'loss': 0.0, 'acc': 0.0}

    returns = {
        'loss': torch.tensor(loss).mean().item(),
        'acc': torch.tensor(acc).mean().item()
    }
    return returns

def estimate_loss(model,*loaders):
    train_loader,val_loader,test_loader = loaders
    returns =  \
        {
            'train': _loss(model, train_loader),
            'val': _loss(model, val_loader),
            'test': _loss(model, test_loader)
        }

    model.train()

    return returns

def train_model(model, optimizer,epoch=10, penalty=False, *loaders):
    train_loader,val_loader,test_loader = loaders
    store_loss = [[],[],[]]  # train val test
    store_acc = [[],[],[]]
    iterations_per_epoch = len(train_loader)
    milestone_epochs = [320000 // iterations_per_epoch, 480000 // iterations_per_epoch]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone_epochs, gamma=0.1)
    scaler = GradScaler('cuda')

    since = time.time()
    for e in range(epoch):
        model.train()
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast('cuda'):
                logits = model(inputs)
                loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        stats = estimate_loss(model,train_loader,val_loader,test_loader)

        str_train_loss = (stats['train']['loss'])
        str_val_loss = (stats['val']['loss'])
        str_test_loss = (stats['test']['loss'])

        train_loss = str_train_loss
        val_loss = str_val_loss
        test_loss = str_test_loss

        train_acc = stats['train']['acc']
        val_acc = stats['val']['acc']
        test_acc = stats['test']['acc']

        now = time.time()
        delta_time = now - since
        since = now

        print \
            (
                f'epoch: {e+1} | time: {delta_time}\n'
                f'train loss: {train_loss}\n'
                f'val loss: {val_loss}\n'
                f'test loss: {test_loss}\n'
                f'train accuracy(训练正确率): {train_acc*100}%\n'
                f'val accuracy(验证正确率): {val_acc*100}%\n'
                f'test accuracy(最终的测试正确率): {test_acc*100}%\n'
            )

        store_loss[0].append(stats['train']['loss'])
        store_acc[0].append(stats['train']['acc'])
        store_loss[1].append(stats['val']['loss'])
        store_acc[1].append(stats['val']['acc'])
        store_loss[2].append(stats['test']['loss'])
        store_acc[2].append(stats['test']['acc'])
        scheduler.step()

    return store_loss,store_acc

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    from Model import *  # load model
    from Data_set import *  # get dataloader
    from Picture.plot import plot  # plot

    CNN = CNN.CNN
    ResNet = ResNet.ResNet
    DResNet = DResNet.DResNet

    classes = None
    name = 'Tiny ImageNet'
    layer = '18'
    epochs = 200
    # train_loader, val_loader, test_loader, classes  # from Data_set/__all__

    print(f"dataset：{name}")

    x = range(epochs)

    model_1 = ResNet().to(device)
    optimizer_1 = optim.AdamW(model_1.parameters(), lr=0.001, weight_decay=0.01)
    store_loss_1, store_acc_1 = train_model(model_1, optimizer_1, epochs, False, train_loader, val_loader, test_loader)

    model_2 = CNN().to(device)
    optimizer_2 = optim.AdamW(model_2.parameters(), lr=0.001, weight_decay=0.01)
    store_loss_2, store_acc_2 = train_model(model_2, optimizer_2, epochs, False, train_loader, val_loader, test_loader)

    plot(x, store_loss_2, kind='loss', model_name=f'CNN_{layer}', layer=layer)
    plot(x, store_acc_2, kind='acc', model_name=f'CNN_{layer}', layer=layer)

    model_3 = DResNet().to(device)
    optimizer_3 = optim.AdamW(model_3.parameters(), lr=0.001, weight_decay=0.01)
    store_loss_3, store_acc_3 = train_model(model_3, optimizer_3, epochs, False, train_loader, val_loader, test_loader)

    plot(x, store_loss_3, kind='loss', model_name=f'DResNet_{layer}', layer=layer)
    plot(x, store_acc_3, kind='acc', model_name=f'DResNet_{layer}', layer=layer)

    # model_4 = DenseNet().to(device)
    # optimizer_4 = optim.SGD(model_4.parameters(), lr=0.1, weight_decay=0.0001, momentum=0.9)
    # store_loss_4, store_acc_4 = train_model(model_4, optimizer_4,epochs,False, train_loader,val_loader,test_loader)
    #
    # plot(x, store_loss_4, kind='loss', model_name=f'DenseNet_{layer}',layer=layer)
    # plot(x, store_acc_4, kind='acc', model_name=f'DenseNet_{layer}',layer=layer)

    color = ['r', 'g', 'b', 'k']
    labels = ['ResNet', 'CNN', 'DResNet', 'DenseNet']
    for idx, y in enumerate((store_acc_1, store_acc_2, store_acc_3)):
        plt.plot(x, y[-1], c=color[idx], label=labels[idx], linewidth=0.7)
    plt.title("acc")
    plt.legend(loc=4)
    plt.savefig(f'./{layer}/acc_{18}.png', dpi=300)
    plt.close()

    for idx, y in enumerate((store_loss_1, store_loss_2, store_loss_3)):
        plt.plot(x, y[-1], c=color[idx], label=labels[idx], linewidth=0.7)
    plt.title("loss")
    plt.legend(loc=1)
    plt.savefig(f'./{layer}/loss_{layer}.png', dpi=300)
    plt.close()

    print(f"the parameter of CNN: {count_parameters(model_2):,}")
    print(f"the parameter of ResNet: {count_parameters(model_1):,}")
    print(f"the parameter of DResNet: {count_parameters(model_3):,}")
    # print(f"the parameter of DenseNet: {count_parameters(model_4):,}")

    df = \
        {
            'name': [f"CNN_{layer}", f"ResNet_{layer}", f"DResNet_{layer}", f"DenseNet_{layer}"],
            'parameters': \
                [
                    f"{count_parameters(model_2):,}",
                    f"{count_parameters(model_1):,}",
                    f"{count_parameters(model_3):,}",
                    # f"{count_parameters(model_4):,}"
                ]
        }
    df = pl.DataFrame(df)
    df.write_csv(f"./{layer}/Parameters.csv")