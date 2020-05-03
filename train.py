import os
import shutil
import argparse

import torch
import torch.nn as nn

import _init_paths  # pylint: disable=unused-import
from pet.utils.misc import mkdir_p, logging_rank
from pet.utils.net import convert_bn2affine_model, convert_conv2syncbn_model, mismatch_params_filter
from pet.utils.checkpointer import CheckPointer
from pet.utils.optimizer import Optimizer
from pet.utils.lr_scheduler import LearningRateScheduler
from pet.utils.logger import TrainingLogger

from pet.rcnn.core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from pet.rcnn.datasets import build_dataset, make_train_data_loader
from pet.rcnn.modeling.model_builder import Generalized_RCNN

# Parse arguments
parser = argparse.ArgumentParser(description='Pet Model Training')
parser.add_argument('--cfg', dest='cfg_file',
                    help='optional config file',
                    default='./cfgs/rcnn/mscoco/e2e_faster_rcnn_R-50-FPN_1x.yaml', type=str)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('opts', help='See pet/rcnn/core/config.py for all options',
                    default=None,
                    nargs=argparse.REMAINDER)

args = parser.parse_args()
if args.cfg_file is not None:
    merge_cfg_from_file(args.cfg_file)
if args.opts is not None:
    merge_cfg_from_list(args.opts)

args.device = torch.device(cfg.DEVICE)
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
args.distributed = num_gpus > 1
if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    args.world_size = torch.distributed.get_world_size()
else:
    args.world_size = 1
    args.local_rank = 0
    cfg.NUM_GPUS = len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) if cfg.DEVICE == 'cuda' else 1
    cfg.TRAIN.LOADER_THREADS *= cfg.NUM_GPUS
    cfg.TEST.LOADER_THREADS *= cfg.NUM_GPUS
    cfg.TEST.IMS_PER_GPU *= cfg.NUM_GPUS

logging_rank('Called with args: {}'.format(args), distributed=args.distributed, local_rank=args.local_rank)


def train(model, loader, optimizer, scheduler, checkpointer, logger):
    # switch to train mode
    model.train()

    # main loop
    start_iter = scheduler.iteration
    loss_ann_tmp = torch.ones([1])
    loss_u_tmp = torch.ones([1])
    loss_v_tmp = torch.ones([1])
    loss_i_tmp = torch.ones([1])
    loss_rpn_objectness_tmp = torch.ones([1])
    loss_rpn_box_reg_tmp = torch.ones([1])
    loss_classifier_tmp = torch.ones([1])
    for iteration, (images, targets, _) in enumerate(loader, start_iter):
        logger.iter_tic()
        logger.data_tic()

        scheduler.step()  # adjust learning rate
        optimizer.zero_grad()

        images = images.to(args.device)
        targets = [target.to(args.device) for target in targets]
        logger.data_toc()

        outputs = model(images, targets)

        logger.update_stats(outputs, args.distributed, args.world_size)

        loss_rpn_objectness = outputs['losses']['loss_rpn_objectness']
        loss_rpn_box_reg = outputs['losses']['loss_rpn_box_reg']
        loss_classifier = outputs['losses']['loss_classifier']
        loss_box_reg = outputs['losses']['loss_box_reg']
        loss_seg_Ann = outputs['losses']['loss_seg_Ann']
        loss_Upoints = outputs['losses']['loss_Upoints']
        loss_Vpoints = outputs['losses']['loss_Vpoints']
        loss_Ipoints = outputs['losses']['loss_IPoints']

        # t_seg_ann = loss_seg_Ann / loss_ann_tmp
        # t_u_points = loss_Upoints / loss_u_tmp
        # t_v_points = loss_Vpoints / loss_v_tmp
        # t_i_points = loss_Ipoints / loss_i_tmp
        # w_seg_ann = loss_seg_Ann * -1.
        # w_u_points = loss_Upoints * -1
        # w_v_points = loss_Vpoints * -1
        # w_i_points = loss_Ipoints * -1
        # w_rpn_objectness = loss_rpn_objectness * -1
        # w_rpn_box_reg = loss_rpn_box_reg * -1
        # w_classifier = loss_classifier * -1
        # w_box_reg = loss_box_reg * -1
        # loss_ann_tmp = loss_seg_Ann.detach()
        # loss_u_tmp = loss_Upoints.detach()
        # loss_v_tmp = loss_Vpoints.detach()
        # loss_i_tmp = loss_Ipoints.detach()
        # print("rpn_objectness:", loss_rpn_objectness.item(), "rpn_box_reg:", loss_rpn_box_reg.item(), \
        #         "classifier:,", loss_classifier.item(), "box_reg:", loss_box_reg.item(), \
        #       "seg_Ann:", loss_seg_Ann.item(), 'Upoints:', loss_Upoints.item(), 'Vpoints:',\
        #       loss_Vpoints.item(), "Ipoints:", loss_Ipoints.item())

        # w_sum = torch.exp(w_seg_ann) + torch.exp(w_u_points) + torch.exp(w_v_points) + torch.exp(w_i_points)
        # torch.exp(w_rpn_objectness)+torch.exp(w_rpn_box_reg)+torch.exp(w_classifier)+torch.exp(w_box_reg)

        # y_seg_ann = (torch.exp(w_seg_ann) / w_sum).item()
        # y_u_points = (torch.exp(w_u_points) / w_sum).item()
        # y_v_points = (torch.exp(w_v_points) / w_sum).item()
        # y_i_points = (torch.exp(w_i_points) / w_sum).item()
        # y_rpn_objectness= torch.exp(w_rpn_objectness) / w_sum
        # y_rpn_box_reg = torch.exp(w_rpn_box_reg) / w_sum
        # y_classifier  = torch.exp(w_classifier) / w_sum
        # y_box_reg = torch.exp(w_box_reg) / w_sum
        # print(y_seg_ann, y_u_points, y_v_points, y_i_points, y_rpn_objectness, y_rpn_box_reg, y_classifier, y_box_reg)
        # with open('./loss.txt', 'a') as f:
        #     f.write(str(y_seg_ann)+' '+str(y_u_points)+ ' '+ str(y_v_points)+' '+str(y_i_points) + '\n')
        # loss = y_rpn_objectness*loss_rpn_objectness + y_classifier*loss_classifier + y_rpn_box_reg*loss_rpn_box_reg + y_box_reg*loss_box_reg+ \
        #        y_seg_ann *loss_seg_Ann+y_u_points*loss_Upoints+y_v_points*loss_Vpoints+y_i_points*loss_Ipoints
        # y_seg_ann = torch.exp(w_seg_ann) / w_sum
        # y_u_points = torch.exp(w_u_points) / w_sum
        # y_v_points = torch.exp(w_v_points) / w_sum
        # y_i_points = torch.exp(w_i_points) / w_sum
        # loss = loss_rpn_objectness + loss_classifier + loss_rpn_box_reg + loss_box_reg + \
        #        1.5*(y_seg_ann * loss_seg_Ann + y_u_points * loss_Upoints + y_v_points * loss_Vpoints + y_i_points * loss_Ipoints)
        # if iteration <= 10000:
        #     alpha = iteration / 10000
        #     warmup_factor_uv = 0.001 * (1 - alpha) + alpha
        #     # warmup_factor_iseg = 0.01 * (1 - alpha) + alpha
        #     loss = loss_rpn_objectness + loss_classifier + loss_rpn_box_reg + loss_box_reg + \
        #            1*(loss_seg_Ann + loss_Ipoints) + warmup_factor_uv*(loss_Upoints + loss_Vpoints)
        # else:
        loss = outputs['total_loss']

        # loss = outputs['total_loss']*scheduler.new_lr*50

        # if iteration % 20 == 0:
        #     print('loss', loss)

        loss.backward()
        optimizer.step()

        if args.local_rank == 0:
            logger.log_stats(scheduler.iteration, scheduler.new_lr)

            # Save model
            if cfg.SOLVER.SNAPSHOT_ITERS > 0 and (iteration + 1) % cfg.SOLVER.SNAPSHOT_ITERS == 0:
                checkpointer.save(model, optimizer, scheduler, copy_latest=True, infix='iter')
        logger.iter_toc()
    if args.local_rank == 0:
        checkpointer.save(model, optimizer, scheduler, copy_latest=True, infix='iter')
    return None


def main():
    if not os.path.isdir(cfg.CKPT):
        mkdir_p(cfg.CKPT)
    if args.cfg_file is not None:
        shutil.copyfile(args.cfg_file, os.path.join(cfg.CKPT, args.cfg_file.split('/')[-1]))
    assert_and_infer_cfg(make_immutable=False)

    # Create model
    model = Generalized_RCNN()
    logging_rank(model, distributed=args.distributed, local_rank=args.local_rank)

    # Create checkpointer
    checkpointer = CheckPointer(cfg.CKPT, weights_path=cfg.TRAIN.WEIGHTS, auto_resume=cfg.TRAIN.AUTO_RESUME,
                                local_rank=args.local_rank)

    # Load model or random-initialization
    model = checkpointer.load_model(model, convert_conv1=cfg.MODEL.CONV1_RGB2BGR)
    if cfg.MODEL.BATCH_NORM == 'freeze':
        model = convert_bn2affine_model(model, merge=not checkpointer.resume)
    elif cfg.MODEL.BATCH_NORM == 'sync':
        model = convert_conv2syncbn_model(model)
    model.to(args.device)

    # Create optimizer
    optimizer = Optimizer(model, cfg.SOLVER, local_rank=args.local_rank).build()
    optimizer = checkpointer.load_optimizer(optimizer)
    logging_rank('The mismatch keys: {}'.format(mismatch_params_filter(sorted(checkpointer.mismatch_keys))),
                 distributed=args.distributed, local_rank=args.local_rank)

    # Create scheduler
    scheduler = LearningRateScheduler(optimizer, cfg.SOLVER, start_iter=0, local_rank=args.local_rank)
    scheduler = checkpointer.load_scheduler(scheduler)

    # Create training dataset and loader
    datasets = build_dataset(cfg.TRAIN.DATASETS, is_train=True, local_rank=args.local_rank)
    train_loader = make_train_data_loader(datasets, is_distributed=args.distributed, start_iter=scheduler.iteration)

    # Create training logger
    training_logger = TrainingLogger(args.cfg_file.split('/')[-1], scheduler=scheduler, log_period=cfg.DISPLAY_ITER)

    # Model Distributed
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
        )
    else:
        model = torch.nn.DataParallel(model)

    # Train
    logging_rank('Training starts.', distributed=args.distributed, local_rank=args.local_rank)
    train(model, train_loader, optimizer, scheduler, checkpointer, training_logger)
    logging_rank('Training done.', distributed=args.distributed, local_rank=args.local_rank)


if __name__ == '__main__':
    main()
