import os
import pdb
import shutil
import time
import numpy as np
import math
import torch
import torch.nn.parallel
import torch.utils.data.distributed

from monai.data import decollate_batch
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather


def adjust_learning_rate(optimizer, epoch, train_config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < train_config.warmup_epochs:
        lr = train_config.optim_lr * epoch / train_config.warmup_epochs
    else:
        lr = train_config.optim_lr * 0.5 * \
            (1. + math.cos(math.pi * (epoch - train_config.warmup_epochs) / (train_config.max_epochs - train_config.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def maybe_print(logger, msg):
    if logger is None:
        print(msg)
    else:
        logger.info(msg)


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args, logger):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    # if hasattr(model, 'module'):
    #     if hasattr(model.module,'training'):
    #         model.module.training = True

    for idx, batch_data in enumerate(loader):
        lr_cur_iter = adjust_learning_rate(optimizer, idx / len(loader) + epoch, args)
        # lr_cur_iter = adjust_learning_rate(optimizer, epoch, args)

        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]

        if "LA_Seg" in args.data_dir:
            data, target = torch.tensor(data), torch.tensor(target)
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            if "LA_Seg" in args.data_dir:
                loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.num_samples)
            else:
                loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0 and idx % 10 == 0:
            maybe_print(
                logger,
                "Epoch {}/{} {}/{}, ".format(epoch, args.max_epochs, idx, len(loader))
              + "loss: {:.4f}, ".format(run_loss.avg)
              + "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg, lr_cur_iter


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None, labels=None, logger=None):
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    with torch.no_grad():
        # if hasattr(model, 'module'):
        #     if hasattr(model.module,'training'):
        #         model.module.training = False

        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)

            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(inputs=data, predictor=model)
                else:
                    logits = model(data)

            # # evaluate in original spacing
            # _, _, _h, _w, _d = target.shape
            # target_shape = (_h, _w, _d )
            # logits = F.interpolate(logits, size=target_shape, mode='trilinear')

            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if not logits.is_cuda:
                target = target.cpu()

            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)

            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)

            if args.distributed:
                if "LA_Seg" in args.data_dir:
                    acc_list, not_nans_list = distributed_all_gather(
                        [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.num_samples
                    )
                else:
                    acc_list, not_nans_list = distributed_all_gather(
                        [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                    )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)

            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if args.rank == 0:
                if labels is not None:
                    each_acc = ','.join([f"{labels[str(i_)]}={acc_.item():.4f}" for i_, acc_ in enumerate(run_acc.avg)])
                else:
                    each_acc = 'None'
                avg_acc = np.mean(run_acc.avg[1:])
                maybe_print(
                    logger,
                    "Val {}/{} {}/{}, ".format(epoch, args.max_epochs, idx, len(loader))
                  + "acc={:.4f}".format(avg_acc) + f"({each_acc})" + ", "
                  + "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
    return run_acc.avg


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None, logger=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    maybe_print(logger, f"Saving checkpoint {filename}")


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
    dataset_props=None,
    logger = None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            maybe_print(logger, f"Writing Tensorboard logs to {args.logdir}")
    if dataset_props is not None:
        labels = dataset_props['labels']
    else:
        labels = None
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    val_best_epoch = 0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        maybe_print(logger, f"rank:{args.rank} {time.ctime()} Epoch:{epoch}")
        epoch_time = time.time()
        train_loss, lr_cur = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args, logger=logger
        )
        if args.rank == 0:
            maybe_print(
                logger, 
                "Final training  {}/{}, ".format(epoch, args.max_epochs - 1)
              + "loss: {:.4f}, ".format(train_loss)
              + "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
                labels=labels,
                logger=logger
            )

            if args.rank == 0:
                if labels is not None:
                    each_acc = ','.join([f"{labels[str(i_)]}={acc_.item():.4f}" for i_, acc_ in enumerate(val_avg_acc)])
                else:
                    each_acc = 'None'
                val_avg_acc = np.mean(val_avg_acc[1:])
                maybe_print(
                    logger,
                    "Final validation  {}/{}, ".format(epoch, args.max_epochs - 1)
                  + "acc={:.4f}".format(val_avg_acc) + f"({each_acc})" + ", "
                  + "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    maybe_print(logger, "new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    val_best_epoch = epoch
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt", logger=logger)
                if b_new_best:
                    maybe_print(logger, "Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        # lr_cur = scheduler.get_last_lr()
        if args.rank == 0 and writer is not None:
            print("lr", lr_cur, epoch)
            writer.add_scalar("lr", lr_cur, epoch)
        # if scheduler is not None:
            # scheduler.step()

    if args.rank == 0:
        maybe_print(logger, f"Training Finished !, Best Accuracy: {val_acc_max} @epoch {val_best_epoch}")

    return val_acc_max
