import argparse
import os
import pdb
import time
import random
import medmnist
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from collections import OrderedDict
from copy import deepcopy
from medmnist import INFO, Evaluator
from tensorboardX import SummaryWriter
from tqdm import trange

from utils import Transform3D
from base_model import BaseModel, ClassificationModel

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def lr_poly(base_lr, iter, max_iter, power=0.9):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, lr, num_stemps):
    lr = lr_poly(lr, i_iter, num_stemps)
    optimizer.param_groups[0]['lr'] = lr
    return lr


class Model_Config():
    def __int__(self):
        pass

def main(args):

    lr = args.lr
    data_flag = args.data_flag
    output_root = args.output_root
    num_epochs = args.num_epochs
    gpu_ids = args.gpu_ids
    batch_size = args.batch_size
    download = args.download
    model_flag = args.model_flag
    as_rgb = args.as_rgb
    model_path = args.model_path
    run = args.run
    root = args.root

    info = INFO[data_flag]
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])
    
    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    # if len(gpu_ids) > 0:
    #     os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_ids[0])

    device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu') 

    output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d%H%M%S"))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    print('==> Preparing data...')
    train_transform = Transform3D(train_augmentation=True)
    eval_transform = Transform3D()

    train_dataset = DataClass(split='train', transform=train_transform, download=download, as_rgb=as_rgb, root=root)
    train_dataset_at_eval = DataClass(split='train', transform=eval_transform, download=download, as_rgb=as_rgb, root=root)
    val_dataset = DataClass(split='val', transform=eval_transform, download=download, as_rgb=as_rgb, root=root)
    test_dataset = DataClass(split='test', transform=eval_transform, download=download, as_rgb=as_rgb, root=root)

    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset_at_eval,
                                batch_size=batch_size,
                                shuffle=False)
    val_loader = data.DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False)

    print('==> Building and training model...')

    if model_flag == 'base_vit':
        config = Model_Config()
        config.pos_type = 'sincos3d'
        config.img_size = [args.img_size, args.img_size, args.img_size]
        config.patch_size = [args.patch_size, args.patch_size, args.patch_size]
        config.hidden_size = 768
        config.intermediate_size = 3072
        config.num_attention_heads = 12
        config.num_key_value_heads = 12
        config.num_hidden_layers = 12
        encoder = BaseModel(config)
        model = ClassificationModel(encoder, n_classes)

        if args.pretrain_path is not None:
            model_dict = torch.load(args.pretrain_path)["state_dict"]
            pretrained_state = {k: v for k, v in model_dict.items() if k.startswith('model')}
            model.load_state_dict(pretrained_state, strict=False)

    model = model.to(device)

    train_evaluator = medmnist.Evaluator(data_flag, 'train', root=root)
    val_evaluator = medmnist.Evaluator(data_flag, 'val',  root=root)
    test_evaluator = medmnist.Evaluator(data_flag, 'test',  root=root)

    criterion = nn.CrossEntropyLoss()

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=True)
        train_metrics = test(model, train_evaluator, train_loader_at_eval, criterion, device, run, output_root)
        val_metrics = test(model, val_evaluator, val_loader, criterion, device, run, output_root)
        test_metrics = test(model, test_evaluator, test_loader, criterion, device, run, output_root)

        print('train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2]) + \
              'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2]) + \
              'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2]))

    if num_epochs == 0:
        return

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    if args.sched == 'CosineLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)
    elif args.sched == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * num_epochs, 0.75 * num_epochs], gamma=0.1)

    logs = ['loss', 'auc', 'acc']
    train_logs = ['train_'+log for log in logs]
    val_logs = ['val_'+log for log in logs]
    test_logs = ['test_'+log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs+val_logs+test_logs, 0)
    
    writer = SummaryWriter(log_dir=os.path.join(output_root, 'Tensorboard_Results'))

    best_auc, best_acc = 0, 0
    best_epoch = 0
    best_model = deepcopy(model)

    global iteration
    iteration = 0

    for epoch in trange(num_epochs):
        if epoch - best_epoch > args.patience:
            break
        train_loss = train(model, train_loader, criterion, optimizer, device, writer)
        val_metrics = test(model, val_evaluator, val_loader, criterion, device, run)

        for i, key in enumerate(val_logs):
            log_dict[key] = val_metrics[i]
        for key, value in log_dict.items():
            writer.add_scalar(key, value, epoch)

        cur_auc = val_metrics[1]
        cur_acc = val_metrics[2]

        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_acc = cur_acc
            best_model = deepcopy(model)

            print('cur_best_epoch:', best_epoch, 'cur_best_metric:', best_auc, best_acc)

        if args.sched == 'poly':
            adjust_learning_rate(optimizer, epoch, args.lr, args.num_epochs)
        else:
            scheduler.step()

    state = {
        'net': model.state_dict(),
    }

    path = os.path.join(output_root, 'best_model.pth')
    torch.save(state, path)

    train_metrics = test(best_model, train_evaluator, train_loader_at_eval, criterion, device, run, output_root)
    val_metrics = test(best_model, val_evaluator, val_loader, criterion, device, run, output_root)
    test_metrics = test(best_model, test_evaluator, test_loader, criterion, device, run, output_root)

    train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
    val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])
    test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])

    log = '%s\n' % (data_flag) + train_log + val_log + test_log + '\n'
    print(log)
    
    with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
        f.write(log)        
            
    writer.close()


def train(model, train_loader, criterion, optimizer, device, writer):
    total_loss = []
    global iteration

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):

        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        targets = torch.squeeze(targets, 1).long().to(device)
        loss = criterion(outputs, targets)

        total_loss.append(loss.item())
        writer.add_scalar('train_loss_logs', loss.item(), iteration)
        iteration += 1

        loss.backward()
        optimizer.step()

    
    epoch_loss = sum(total_loss)/len(total_loss)
    return epoch_loss


def test(model, evaluator, data_loader, criterion, device, run, save_folder=None):

    model.eval()

    total_loss = []
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))

            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)
            m = nn.Softmax(dim=1)
            outputs = m(outputs).to(device)
            targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())

            y_score = torch.cat((y_score, outputs), 0)

        y_score = y_score.detach().cpu().numpy()
        auc, acc = evaluator.evaluate(y_score, save_folder, run)

        test_loss = sum(total_loss) / len(total_loss)

        return [test_loss, auc, acc]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RUN Baseline model')

    parser.add_argument('--data_flag',
                        default='nodulemnist3d',
                        type=str)
    parser.add_argument('--num_epochs',
                        default=100,
                        help='num of epochs of training, the script would only test model if set num_epochs to 0',
                        type=int)
    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)
    parser.add_argument('--random_seed',
                        default=42,
                        type=int)
    parser.add_argument('--batch_size',
                        default=16,
                        type=int)
    parser.add_argument('--download',
                        action="store_true")
    parser.add_argument('--as_rgb',
                        help='to copy channels',
                        action="store_true")
    parser.add_argument('--model_path',
                        default=None,
                        help='root of the pretrained model to test',
                        type=str)
    parser.add_argument('--model_flag',
                        default='base_vit',
                        type=str)
    parser.add_argument('--run',
                        default='model1',
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)
    parser.add_argument('--root',
                        default='',
                        help='input root',
                        type=str)
    parser.add_argument('--output_root',
                        default='',
                        help='output root, where to save models',
                        type=str)
    parser.add_argument("--pretrain_path", default=None, type=str)
    parser.add_argument("--img_size", default=128, type=int)
    parser.add_argument("--patch_size", default=16, type=int)
    parser.add_argument("--patience", default=30, type=int)
    parser.add_argument("--lr", default=1.5e-5, type=float, help="learning rate")
    parser.add_argument('--optim', default='adamw', type=str)
    parser.add_argument('--sched', default='CosineLR', type=str)

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.cuda.empty_cache()
    print(args)
    main(args)
