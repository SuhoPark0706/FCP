r""" Visual Prompt Encoder training (validation) code """
import os
import argparse

import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributed as dist

from model.VRP_encoder import VRP_encoder
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator, Evaluator_pseudo
from common import utils
from data.dataset import FSSDataset
from SAM2pred import SAM_pred

relu=nn.ReLU()

def l2norm(x):
    """L2-normalize columns of x"""
    return F.normalize(x, p=2, dim=-1)

def cal_sim_loss(c_attn_map):
    c_attn_map = l2norm(c_attn_map)
    cur_loss = ( c_attn_map @ c_attn_map.permute(0,2,1) - torch.eye(c_attn_map.shape[1]).cuda() )
    cur_loss = (relu(cur_loss)).mean()
    return cur_loss

def train(args, epoch, model, sam_model, dataloader, optimizer, scheduler, training):
    r""" Train VRP_encoder model """

    utils.fix_randseed(args.seed + epoch) if training else utils.fix_randseed(args.seed)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)
    average_meter_pseudo = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        
        batch = utils.to_cuda(batch)

        query_sam_feat = sam_model.get_feat_from_np(batch['query_img'], batch['query_name'], torch.tensor([0]).cuda())
        supp_sam_feat = sam_model.get_feat_from_np(batch['support_imgs'].squeeze(1), batch['support_names'][0], torch.tensor([0]).cuda())

        protos, attn_lst, dict_for_loss = model(args.condition, query_sam_feat, supp_sam_feat, batch['query_img'], batch['query_mask'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1), training)

        """calculate pseudo mask loss"""
        query_mask = F.interpolate(batch['query_mask'].unsqueeze(1).float(), size=(64,64), mode='nearest').squeeze(1)

        pseudo_masks = dict_for_loss['pseudo_mask_loss']
        pseudo_mask_loss = 0
        for m, msk in enumerate(pseudo_masks):
            pseudo_mask_loss = pseudo_mask_loss + model.module.compute_pseudo_objective(msk, query_mask)/(len(pseudo_masks)-m)
            
        """sim_loss"""
        query_sam_c_attn = dict_for_loss['query_sam_c_attn']
        support_sam_c_attn = dict_for_loss['support_sam_c_attn']
        q_attn_loss = 0
        for attn_map in query_sam_c_attn:
            q_attn_loss = q_attn_loss + cal_sim_loss(attn_map)
            
        s_attn_loss = 0
        for attn_map in support_sam_c_attn:
            s_attn_loss = s_attn_loss + cal_sim_loss(attn_map)

        attn_loss = q_attn_loss + s_attn_loss
    
        """calculate prompt loss"""
        logit_mask, pred_mask = sam_model(batch['query_img'], batch['query_name'], protos)
        pred_mask = (torch.sigmoid(logit_mask) > 0.5).float()
        prompt_loss = model.module.compute_objective(logit_mask, batch['query_mask'])

        """add loss"""
        loss = prompt_loss*args.prompt_loss + pseudo_mask_loss*args.mask_loss + attn_loss*args.attn_loss
        
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        area_inter, area_union = Evaluator.classify_prediction(pred_mask.squeeze(1), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)
        
        pseudo_masks[-1] = (pseudo_masks[-1] > 0.5).float()       
        area_inter_pseudo, area_union_pseudo, area_pred_pseudo, area_gt_pseudo = Evaluator_pseudo.classify_prediction(pseudo_masks[-1].squeeze(1), query_mask, batch)
        average_meter_pseudo.update(area_inter_pseudo, area_union_pseudo, batch['class_id'], loss.detach().clone(), pred=area_pred_pseudo, gt=area_gt_pseudo)
        average_meter_pseudo.write_process(idx, len(dataloader), epoch, write_batch_idx=50, miou_only=True)
        
    average_meter.write_result('Training' if training else 'Validation', epoch)
    average_meter_pseudo.write_result('Training' if training else 'Validation', epoch, miou_only=True)
    
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()
    miou_pseudo, _ = average_meter_pseudo.compute_iou()

    return avg_loss, miou, fb_iou, miou_pseudo


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Visual Prompt Encoder Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='/root/paddlejob/workspace/env_run/datsets/')
    parser.add_argument('--benchmark', type=str, default='coco', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=2) # batch size = num_gpu * bsz default num_gpu = 4
    parser.add_argument('--prompt_loss', type=float, default=0)
    parser.add_argument('--mask_loss', type=float, default=0.5)
    parser.add_argument('--attn_loss', type=float, default=0.5)
    parser.add_argument('--attn_drop_out', type=float, default=0.3)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--condition', type=str, default='scribble', choices=['point', 'scribble', 'box', 'mask'])
    parser.add_argument('--use_ignore', type=bool, default=True, help='Boundaries are not considered during pascal training')
    parser.add_argument('--local_rank', type=int, default=-1, help='number of cpu threads to use during batch generation')
    parser.add_argument('--num_query', type=int, default=50)
    parser.add_argument('--spt_num_query', type=int, default=50)
    parser.add_argument('--concat_th', type=bool, default=False)
    parser.add_argument('--use_log', action='store_true')
    parser.add_argument('--load', type=str, default="./dummy.pt")
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
    args = parser.parse_args()
    # Distributed setting
    local_rank = args.local_rank
    dist.init_process_group(backend='nccl')
    print('local_rank: ', local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    
    
    if utils.is_main_process():
        Logger.initialize(args, training=args.use_log)
    utils.fix_randseed(args.seed)
    # Model initialization
    model = VRP_encoder(args, args.backbone, False)
    if utils.is_main_process():
        Logger.log_params(model)

    sam_model = SAM_pred()
    sam_model.to(device)
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Device setup
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    for param in model.module.layer0.parameters():
        param.requires_grad = False
    for param in model.module.layer1.parameters():
        param.requires_grad = False
    for param in model.module.layer2.parameters():
        param.requires_grad = False
    for param in model.module.layer3.parameters():
        param.requires_grad = False
    for param in model.module.layer4.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW([
        {'params': model.module.downsample_query.parameters(), "lr": args.lr},
        {'params': model.module.downsample_sam_query.parameters(), "lr": args.lr},
        {'params': model.module.transformer_decoder.parameters()},
        ],lr = args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    Evaluator.initialize(args)
    Evaluator_pseudo.initialize(args)

    # Dataset initialization
    FSSDataset.initialize(img_size=512, datapath=args.datapath, use_original_imgsize=False)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')

    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= args.epochs * len(dataloader_trn))
    # Training 
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(args.epochs):

        trn_loss, trn_miou, trn_fb_iou, miou_pseudo = train(args, epoch, model, sam_model, dataloader_trn, optimizer, scheduler, training=True)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou, miou_pseudo = train(args, epoch, model, sam_model, dataloader_val, optimizer, scheduler, training=False)

        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            if utils.is_main_process():
                Logger.save_model_miou(model, epoch, val_miou)
        if utils.is_main_process():
            Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
            Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
            Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
            Logger.tbd_writer.flush()
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')