import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from timm.models import create_model

from mesh.dataset.kinetics import build_pretraining_dataset
from mesh.dataset.utils import ReshapeVideo
from mesh.msc.engine_for_pretraining import (ENABLE_FILTERING,
                                             ENABLE_MULTILAYER_OUTPUT,
                                             train_one_epoch)
from mesh.msc.loss import loss_function
from mesh.msc.utils import (DDP_PARAMETER_CHECK, NativeScalerWithGradNormCount,
                            TensorboardLogger, auto_load_model,
                            check_world_size, close_distributed_mode,
                            cosine_scheduler, get_rank, get_world_size,
                            init_distributed_mode, is_main_process, save_model,
                            seed_worker)
from mesh.vitransformer.arguments import get_args
from mesh.vitransformer.optim_factory import create_optimizer
from mesh.vitransformer.pretrain import __all__

WORLD_SIZE, VISIBLE_IDS = check_world_size()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        decoder_depth=args.decoder_depth,
        mask_ratio=args.mask_ratio,
        decoder_eval=False
    )
    return model


def main(rank, args):
    init_distributed_mode(WORLD_SIZE, VISIBLE_IDS, rank, args)

    print(args)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size //
                        patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    # get dataset
    dataset_train = build_pretraining_dataset(args)

    num_tasks = get_world_size()
    global_rank = get_rank()
    sampler_rank = global_rank
    num_training_steps_per_epoch = len(
        dataset_train) // args.batch_size // num_tasks

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=seed_worker
    )

    model.to(rank)
    model_without_ddp = model
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params: {} M'.format(n_parameters / 1e6))

    total_batch_size = args.batch_size * get_world_size()

    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" %
          (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=DDP_PARAMETER_CHECK)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp)
    loss_scaler = NativeScalerWithGradNormCount()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" %
          (max(wd_schedule_values), min(wd_schedule_values)))

    auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    torch.cuda.empty_cache()
    print(f"Start training for {args.epochs} epochs")
    print(f"ENABLE_FILTERING={ENABLE_FILTERING}")
    print(f"ENABLE_MULTILAYER_OUTPUT={ENABLE_MULTILAYER_OUTPUT}")

    loss_func = loss_function(args.loss_index, WORLD_SIZE, rank)
    rev = ReshapeVideo(args.num_frames)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        try:
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            if log_writer is not None:
                log_writer.set_step(epoch * num_training_steps_per_epoch)
            train_stats = train_one_epoch(
                args.loss_index, loss_func, rev, model, data_loader_train,
                args.acc_target, optimizer, rank, epoch, loss_scaler,
                args.clip_grad, log_writer=log_writer,
                start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values,
                wd_schedule_values=wd_schedule_values,
                num_frames=args.num_frames,
                input_size=args.input_size,
                patch_size=patch_size[0],
                normlize_target=args.normlize_target
            )
        except Exception as e:
            raise
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    close_distributed_mode(args)


if __name__ == '__main__':
    opts = get_args()
    if opts.loss_index == 1:
        opts.output_dir += "/1_vision"
    else:
        opts.batch_size = 1
        opts.output_dir += ("/"+str(opts.loss_index)+"_accuracy")
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    if WORLD_SIZE == 1:
        main(VISIBLE_IDS[0], opts)
    else:
        try:
            mp.spawn(main, args=(opts,), nprocs=WORLD_SIZE)
        except Exception as e:
            print(e)
            if "DataLoader" in str(e):
                print('[Set persistent_workers to True in DataLoader!]')
