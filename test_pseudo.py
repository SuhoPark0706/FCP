r""" Visual Prompt Encoder training (validation) code """
import os
import argparse

import torch.nn as nn
import torch
import torch.distributed as dist

from model.VRP_encoder import VRP_encoder
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator_for_pseudo
from common import utils
from data.dataset import FSSDataset
from SAM2pred import SAM_pred

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def test(args, epoch, model, sam_model, dataloader, training):
    r""" Train VRP_encoder model """
    training = False
    utils.fix_randseed(args.seed + epoch) if training else utils.fix_randseed(args.seed)
    model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        
        batch = utils.to_cuda(batch)
        
        query_sam_feat = sam_model.get_feat_from_np(batch['query_img'], batch['query_name'], torch.tensor([0]).cuda())
        supp_sam_feat = sam_model.get_feat_from_np(batch['support_imgs'].squeeze(1), batch['support_names'][0], torch.tensor([0]).cuda())
        
        protos, attn_lst, pseudo_masks, pseudo_mask_vis = model(args.condition, query_sam_feat, supp_sam_feat, batch['query_img'], batch['query_mask'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1), training)
        low_masks, pred_mask = sam_model(batch['query_img'], batch['query_name'], protos)
        logit_mask = low_masks
        
        query_mask = F.interpolate(batch['query_mask'].unsqueeze(1).float(), size=(64,64), mode='nearest').squeeze(1)
        
        pred_mask = torch.sigmoid(logit_mask) > 0.5
        pred_mask = pred_mask.float()

        pseudo_masks[-1] = (pseudo_masks[-1] > 0.5).float()

        loss = model.module.compute_objective(logit_mask, batch['query_mask'])

        area_inter, area_union = Evaluator_for_pseudo.classify_prediction(pseudo_masks[-1].squeeze(1), query_mask)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)
        
        # # Visualize predictions
        # if Visualizer.visualize:
        #     Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
        #                                           batch['query_img'], batch['query_mask'],
        #                                           pred_mask, batch['class_id'], idx,
        #                                           area_inter[1].float() / area_union[1].float())
        
        # """visualization"""
        # img_size = batch['query_img'].shape
        # spt_img = batch['support_imgs'].squeeze(1).squeeze(0).cpu().numpy()
        # spt_img = np.transpose(spt_img, (1, 2, 0))
        # qry_img = batch['query_img'].cpu().squeeze(0).numpy()
        # qry_img = np.transpose(qry_img, (1, 2, 0))
        
        # """show attn map"""
        # target_img = spt_img
        # target_attn_map = attn_lst[0]
        # dir_path = './vis_s_attn'
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        # for j in range(target_attn_map.shape[1]):
        #     attn = target_attn_map[:,j,:]
        #     attn = (attn - attn.min())/(attn.max() - attn.min())
        #     attn = attn.reshape(64,64)
        #     attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), size=img_size[2:], mode='bilinear', align_corners=True).cpu().squeeze(0).squeeze(0).numpy()

        #     cmap = plt.get_cmap('jet')
        #     attn_colormap = cmap(attn)
        #     attn_colormap = attn_colormap[..., :3]
            
        #     alpha = 0.8
        #     combined_img = (1 - alpha) * target_img + alpha * attn_colormap
            
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(combined_img)
        #     plt.axis('off')  # 축 제거
        #     plt.savefig(dir_path + '/{}_{}.png'.format(idx,j), bbox_inches='tight', pad_inches=0)
        #     plt.close()
            
        # """show attn map"""
        # target_img = qry_img
        # target_attn_map = attn_lst[1]
        # dir_path = './vis_q_attn'
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        # for j in range(target_attn_map.shape[1]):
        #     attn = target_attn_map[:,j,:]
        #     attn = (attn - attn.min())/(attn.max() - attn.min())
        #     attn = attn.reshape(64,64)
        #     attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), size=img_size[2:], mode='bilinear', align_corners=True).cpu().squeeze(0).squeeze(0).numpy()

        #     cmap = plt.get_cmap('jet')
        #     attn_colormap = cmap(attn)
        #     attn_colormap = attn_colormap[..., :3]
            
        #     alpha = 0.8
        #     combined_img = (1 - alpha) * target_img + alpha * attn_colormap
            
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(combined_img)
        #     plt.axis('off')  # 축 제거
        #     plt.savefig(dir_path + '/{}_{}.png'.format(idx,j), bbox_inches='tight', pad_inches=0)
        #     plt.close()
        
        # """show attn map"""
        # target_img = spt_img
        # target_attn_map = attn_lst[0]
        # dir_path = './vis_s_c_attn'
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        # for j in range(target_attn_map.shape[1]):
        #     attn = target_attn_map[:,j,:]
        #     attn = (attn - attn.min())/(attn.max() - attn.min())
        #     attn = attn.reshape(64,64)
        #     attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), size=img_size[2:], mode='bilinear', align_corners=True).cpu().squeeze(0).squeeze(0).numpy()

        #     cmap = plt.get_cmap('jet')
        #     attn_colormap = cmap(attn)
        #     attn_colormap = attn_colormap[..., :3]
            
        #     alpha = 0.8
        #     combined_img = (1 - alpha) * target_img + alpha * attn_colormap
            
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(combined_img)
        #     plt.axis('off')  # 축 제거
        #     plt.savefig(dir_path + '/{}_{}.png'.format(idx,j), bbox_inches='tight', pad_inches=0)
        #     plt.close()
            
        # """show attn map"""
        # target_img = qry_img
        # target_attn_map = attn_lst[1]
        # dir_path = './vis_q_c_attn'
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        # for j in range(target_attn_map.shape[1]):
        #     attn = target_attn_map[:,j,:]
        #     attn = (attn - attn.min())/(attn.max() - attn.min())
        #     attn = attn.reshape(64,64)
        #     attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), size=img_size[2:], mode='bilinear', align_corners=True).cpu().squeeze(0).squeeze(0).numpy()

        #     cmap = plt.get_cmap('jet')
        #     attn_colormap = cmap(attn)
        #     attn_colormap = attn_colormap[..., :3]
            
        #     alpha = 0.8
        #     combined_img = (1 - alpha) * target_img + alpha * attn_colormap
            
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(combined_img)
        #     plt.axis('off')  # 축 제거
        #     plt.savefig(dir_path + '/{}_{}.png'.format(idx,j), bbox_inches='tight', pad_inches=0)
        #     plt.close()
            
        # """show pseudo mask"""
        # for k in range(len(pseudo_mask_vis)):
        #     target_img = qry_img
        #     target_attn_map = pseudo_mask_vis[k]
        #     dir_path = './vis_pseudo_mask'
        #     if not os.path.exists(dir_path):
        #         os.makedirs(dir_path)
                
        #     attn = target_attn_map[:,:]
        #     attn = (attn - attn.min())/(attn.max() - attn.min())
        #     attn = attn.reshape(64,64)
        #     attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), size=img_size[2:], mode='bilinear', align_corners=True).cpu().squeeze(0).squeeze(0).numpy()

        #     cmap = plt.get_cmap('jet')
        #     attn_colormap = cmap(attn)
        #     attn_colormap = attn_colormap[..., :3]
            
        #     alpha = 0.8
        #     combined_img = (1 - alpha) * target_img + alpha * attn_colormap
            
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(combined_img)
        #     plt.axis('off')  # 축 제거
        #     plt.savefig(dir_path + '/{}_{}.png'.format(idx, k), bbox_inches='tight', pad_inches=0)
        #     plt.close()

    average_meter.write_result('Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Visual Prompt Encoder Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='/root/paddlejob/workspace/env_run/datsets/')
    parser.add_argument('--benchmark', type=str, default='coco', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=2) # batch size = num_gpu * bsz default num_gpu = 4
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--supp_ratio', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--condition', type=str, default='scribble', choices=['point', 'scribble', 'box', 'mask'])
    parser.add_argument('--use_ignore', type=bool, default=True, help='Boundaries are not considered during pascal training')
    parser.add_argument('--num_query', type=int, default=50)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--local_rank', type=int, default=-1, help='number of cpu threads to use during batch generation')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--load', type=str, default="./best_model.pt")
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--visualize', type=bool, default=True, help='Boundaries are not considered during pascal training')
    parser.add_argument('--vispath', type=str, default='./vis')    
    args = parser.parse_args()

    Logger.initialize(args, training=False)
    
    # Distributed setting
    local_rank = args.local_rank
    dist.init_process_group(backend='nccl')
    print('local_rank: ', local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
        
    # Model initialization
    model = VRP_encoder(args, args.backbone, False)
    model.eval()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    # model = nn.DataParallel(model)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    sam_model = SAM_pred()
    sam_model.to(device)
    model.to(device)
    
    # Load trained model
    if args.load == '': raise Exception('Pretrained model not specified.')
    model.load_state_dict(torch.load(args.load))

    # Helper classes (for testing) initialization
    Evaluator_for_pseudo.initialize(args)
    Visualizer.initialize(args.visualize, "./vis/")

    # Dataset initialization
    FSSDataset.initialize(img_size=512, datapath=args.datapath, use_original_imgsize=False)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')

    # Test
    with torch.no_grad():
        test_miou, test_fb_iou = test(args, 0, model, sam_model, dataloader_test, False)
    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
