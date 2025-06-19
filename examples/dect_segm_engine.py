import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, lr_scheduler = None, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # lr_scheduler is 'default' (following torch tutorial), None (constant lr), or GeN_lr (ours)
    if epoch == 0 and lr_scheduler == 'default':
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(data_loader) - 1)
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )
            
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()

            if lr_scheduler=='GeN_lr':
                import numpy as np
                existing_lr = optimizer.param_groups[0]['lr']

                for param in model.parameters():
                    if param.requires_grad:
                        param.previous=param.data.clone()
                optimizer.step()
                for param in model.parameters():
                    if param.requires_grad:
                        param.phi=(param.previous-param.data)/existing_lr
                

                with torch.no_grad():
                    lr_list = np.array([-1, 1,0])*existing_lr
                    loss_list = np.zeros(len(lr_list))

                    for j,lr in enumerate(lr_list):
                        for param in model.parameters():
                            if param.requires_grad:
                              param.data=param.previous-lr*param.phi

                        loss_dict_temp = model(images, targets)
                        # reduce losses over all GPUs for logging purposes
                        loss_dict_reduced_temp = utils.reduce_dict(loss_dict_temp)
                        losses_reduced_temp = sum(loss for loss in loss_dict_reduced_temp.values())

                        loss_temp = losses_reduced_temp.item()

                        loss_list[j]+=loss_temp

                    from scipy.optimize import curve_fit
                    from GeN import fit_func2

                    try:
                        [gHg, Gg] = curve_fit(fit_func2, lr_list,loss_list-loss_list[-1])[0]
                        smooth=0.9
                        if gHg>0 and Gg>0:
                            for g in optimizer.param_groups:
                              g['lr'] = max(min(Gg/gHg,g['lr']*2),g['lr']/2)*(1-smooth)+smooth*g['lr']
                    except:
                        for g in optimizer.param_groups:
                          g['lr'] = g['lr']/2
                        pass
                        
            optimizer.step()

            if not hasattr(model,'lr_history'):
                model.lr_history=[]
                model.loss_history=dict()
            model.lr_history.append(optimizer.param_groups[0]['lr'])
            for key,value in loss_dict_reduced.items():
                if key in model.loss_history:
                    model.loss_history[key]=[value]
                else:
                    model.loss_history[key].append(value)

        if lr_scheduler not in [None, 'GeN_lr']:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
