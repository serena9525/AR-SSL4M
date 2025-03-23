import argparse
import os
import math
import random
import torch
import timeit
import time
import numpy as np
import os.path as osp

# from apex import amp
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn import metrics
from functools import partial

from dataloader import RICORD_Dataset
from utils.ParaFlop import print_model_parm_nums
from utils.utils import ROC
from utils.utils import LayerDecayValueAssigner, get_parameter_groups
from engine import Engine
from base_model import BaseModel, ClassificationModel
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Downstream tasks")

    parser.add_argument("--data_path", type=str, default='')
    parser.add_argument("--snapshot_dir", type=str, default='snapshots/tmp/')
    parser.add_argument("--reload_from_pretrained", type=str2bool, default=False)
    parser.add_argument("--pretrained_path", type=str, default='../snapshots/xx/checkpoint.pth')

    parser.add_argument("--input_size", type=str, default="64,128,128")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--learning_rate_base", type=float, default=1e-5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--val_only", type=int, default=0)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--optim", type=str, default='adamw')
    parser.add_argument('--sched', type=str, default='poly')
    parser.add_argument("--gpu", type=str, default='None')
    parser.add_argument("--arch", type=str, default='base_vit')
    parser.add_argument("--optimal_th", type=float, default=0.5)
    parser.add_argument("--layer_decay", type=float, default=0.75)
    return parser


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate_poly(optimizer, i_iter, lr, num_stemps, power):
    lr = lr_poly(lr, i_iter, num_stemps, power)
    # optimizer.param_groups[0]['lr'] = lr

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr

    return lr


class Model_Config():
    def __int__(self):
        pass


def main():
    """Create the model and start the training."""
    parser = get_arguments()
    start = timeit.default_timer()
    os.environ["OMP_NUM_THREADS"] = "1"

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        print(args)
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        # if not args.gpu == 'None':
        #     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        torch.use_deterministic_algorithms(True)

        # fix random seeds write an function
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.cuda.empty_cache()


        d, h, w = map(int, args.input_size.split(','))
        input_size = (h, w, d)

        if args.arch == 'base_vit':
            config = Model_Config()
            config.img_size = [h, w, d]
            config.patch_size = [16, 16, 16]
            config.pos_type = 'sincos3d'
            config.hidden_size = 768
            config.intermediate_size = 3072
            config.num_attention_heads = 12
            config.num_key_value_heads = 12
            config.num_hidden_layers = 12
            encoder = BaseModel(config)
            model = ClassificationModel(encoder, 2)

            if args.reload_from_pretrained:
                model_dict = torch.load(args.pretrained_path)["state_dict"]
                pretrained_state = {k: v for k, v in model_dict.items() if k.startswith('model')}
                model.load_state_dict(pretrained_state, strict=False)
        else:
            exit()

        print_model_parm_nums(model)
        model.train()

        device = torch.device('cuda:{}'.format(args.local_rank))
        model.to(device)

        if not args.reload_from_pretrained:
            args.learning_rate = args.learning_rate_base
            optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate, weight_decay=0.0001)
        else:
            num_layers = 12
            args.beta1 = 0.9
            args.beta2 = 0.95
            args.weight_decay = 0.05
            assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
            optim_params = get_parameter_groups(args, model,
                                                get_layer_id=partial(assigner.get_layer_id, prefix='model.'),
                                                get_layer_scale=assigner.get_scale,
                                                verbose=False)
            optimizer = torch.optim.AdamW(optim_params,
                                          lr=args.learning_rate,
                                          betas=(args.beta1, args.beta2),
                                          weight_decay=args.weight_decay)

        if args.FP16:
            print("Note: Using FP16 during training************")
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        if args.FP16:
            print("Using FP16 for training!!!")
            scaler = torch.cuda.amp.GradScaler()

        if args.num_gpus > 1:
            model = engine.data_parallel(model)

        snapshot_dir = f'{args.snapshot_dir}{time.strftime("%y%m%d%H%M%S")}/'
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)

        trainloader, train_sampler = engine.get_train_loader(
            RICORD_Dataset(args.data_path, args, list_path="RICORD_train.txt", crop_size_3D=input_size, split="train"), drop_last=True)
        valloader, val_sampler = engine.get_test_loader(
            RICORD_Dataset(args.data_path, args, list_path="RICORD_val.txt", crop_size_3D=input_size, split="val"), batch_size=1)
        testloader, test_sampler = engine.get_test_loader(
            RICORD_Dataset(args.data_path, args, list_path="RICORD_test.txt", crop_size_3D=input_size, split="test"), batch_size=1)

        print("train dataset len: {}, val dataset len: {}".format(len(trainloader), len(valloader)))
        all_tr_loss = []
        best_acc = -1
        optimal_th = args.optimal_th

        for epoch in range(args.start_epoch, args.num_epochs):

            if args.val_only == 1:
                break

            time_t1 = time.time()

            if engine.distributed:
                train_sampler.set_epoch(epoch)

            epoch_loss = []

            if args.sched == 'poly':
                adjust_learning_rate_poly(optimizer, epoch, args.learning_rate, args.num_epochs, args.power)

            model.train()
            model.cal_acc = False
            for iter, (input_ids, labels) in enumerate(trainloader):

                input_ids = input_ids.cuda(non_blocking=True)
                labels = labels.long().cuda(non_blocking=True)

                data = {"data": input_ids, "labels": labels}
                optimizer.zero_grad()

                if args.FP16:
                    with autocast():
                        term_all = model(data)
                        del data

                        term_all = engine.all_reduce_tensor(term_all)

                    scaler.scale(term_all).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    model.cal_acc = False
                    cls_loss = model(data)
                    term_all = cls_loss

                    del data

                    reduce_all = engine.all_reduce_tensor(term_all)

                    term_all.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                    optimizer.step()

                epoch_loss.append(float(reduce_all))

            epoch_loss = np.mean(epoch_loss)
            all_tr_loss.append(epoch_loss)
            time_t2 = time.time()

            if (args.local_rank == 0):
                print('Epoch_sum {}: lr = {:.4}, loss_Sum = {:.4}, time_cost = {}s'.format
                      (epoch, optimizer.param_groups[0]['lr'], epoch_loss.item(), int(time_t2 - time_t1)))


            if args.local_rank == 0:
                model.eval()
                model.cal_acc = True
                pre_score = []
                label_val = []
                epoch_acc = []

                pre_score_test = []
                label_val_test = []
                test_acc = []

                with torch.no_grad():
                    for iter, (input_ids, labels) in enumerate(valloader):
                        input_ids = input_ids.cuda(non_blocking=True)
                        labels = labels.long().cuda(non_blocking=True)
                        data = {"data": input_ids, "labels": labels}
                        term_acc, pred_softmax = model(data)
                        epoch_acc.append(term_acc)
                        pre_score.append(pred_softmax[:, 1].cpu().numpy())
                        label_val.append(labels.cpu().numpy())

                pre_score = np.concatenate(pre_score, 0)
                label_val = np.concatenate(label_val, 0)
                val_auc = metrics.roc_auc_score(label_val, pre_score)

                pre_score[pre_score >= 0.5] = 1
                pre_score[pre_score < 0.5] = 0

                val_f1 = metrics.f1_score(label_val, pre_score)

                # epoch_acc_mean = np.mean(epoch_acc)
                epoch_acc_mean = metrics.accuracy_score(label_val, pre_score)

                if best_acc < (epoch_acc_mean + val_auc):
                    best_acc = epoch_acc_mean + val_auc
                    print(f"save best weight: acc:{epoch_acc_mean}, auc: {val_auc}, f1: {val_f1}")
                    save_dict = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch + 1,
                    }
                    torch.save(save_dict, osp.join(snapshot_dir, 'checkpoint.pth'))

                if epoch == args.num_epochs - 1:
                    with torch.no_grad():
                        for iter, (input_ids, labels) in enumerate(testloader):
                            input_ids = input_ids.cuda(non_blocking=True)
                            labels = labels.long().cuda(non_blocking=True)
                            data = {"data": input_ids, "labels": labels}
                            term_acc, pred_softmax = model(data)
                            test_acc.append(term_acc)
                            pre_score_test.append(pred_softmax[:, 1].cpu().numpy())
                            label_val_test.append(labels.cpu().numpy())

                    pre_score_test = np.concatenate(pre_score_test, 0)
                    label_val_test = np.concatenate(label_val_test, 0)
                    test_auc = metrics.roc_auc_score(label_val_test, pre_score_test)

                    pre_score_test[pre_score_test >= optimal_th] = 1
                    pre_score_test[pre_score_test < optimal_th] = 0

                    test_f1 = metrics.f1_score(label_val_test, pre_score_test)

                    # test_acc_mean = np.mean(test_acc)
                    test_acc_mean = metrics.accuracy_score(label_val_test, pre_score_test)
                    print("last epoch test dataset acc: {}, auc: {}, f1: {}".format(test_acc_mean, test_auc, test_f1))
                    with open(os.path.join(snapshot_dir, "result.txt"), "w") as fp:
                        fp.write("last epoch test dataset acc: {}, auc: {}, f1: {}".format(test_acc_mean, val_auc, val_f1))

        model.eval()
        print("load best weight from", osp.join(snapshot_dir, 'checkpoint.pth'))
        best_performance_weight = torch.load(osp.join(snapshot_dir, 'checkpoint.pth'))['model']
        model.load_state_dict(best_performance_weight, strict=True)
        model.cal_acc = True
        test_acc = []
        pre_score = []
        label_val = []
        with torch.no_grad():
            for iter, (input_ids, labels) in enumerate(testloader):
                input_ids = input_ids.cuda(non_blocking=True)
                labels = labels.long().cuda(non_blocking=True)
                data = {"data": input_ids, "labels": labels}
                term_acc, pred_softmax = model(data)
                test_acc.append(term_acc)
                pre_score.append(pred_softmax[:, 1].cpu().numpy())
                label_val.append(labels.cpu().numpy())

        pre_score = np.concatenate(pre_score, 0)
        label_val = np.concatenate(label_val, 0)

        val_auc = metrics.roc_auc_score(label_val, pre_score)
        pre_score[pre_score >= optimal_th] = 1
        pre_score[pre_score < optimal_th] = 0
        val_f1 = metrics.f1_score(label_val, pre_score)

        # test_acc_mean = np.mean(test_acc)
        test_acc_mean = metrics.accuracy_score(label_val, pre_score)

        print("best epoch test dataset acc: {}, auc: {}, f1: {}".format(test_acc_mean, val_auc, val_f1))
        with open(os.path.join(snapshot_dir, "result.txt"), "a") as fp:
            fp.write("best epoch test dataset acc: {}, auc: {}, f1: {}".format(test_acc_mean, val_auc, val_f1))
        end = timeit.default_timer()
        print(end - start, 'seconds')


if __name__ == '__main__':
    main()
