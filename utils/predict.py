import os
import torch
import torch.nn as nn
from third_party.softsplat import softsplat
import numpy as np
import cv2
from copy import deepcopy

def feature_flow(ft, forward,f_size):
    if forward == False:
        ft = torch.flip(ft, dims=[0])
    trg_ft = nn.Upsample(size=(f_size[0], f_size[1]), mode='bilinear')(ft[1:])
    num_channel = ft.size(1)
    cos = nn.CosineSimilarity(dim=1)
    flow = torch.zeros(f_size[0],f_size[1],2)
    for i in range(f_size[0]):
        for j in range(f_size[1]):
            src_ft = nn.Upsample(size=(f_size[0], f_size[1]), mode='bilinear')(ft[:1])
            src_vec = src_ft[0, :, i, j].view(1, num_channel, 1, 1)  # 1, C, 1, 1
            cos_map = cos(src_vec, trg_ft).cpu().numpy()
            max_yx = np.unravel_index(cos_map[0].argmax(), cos_map[0].shape)
            flow[i,j,0]=max_yx[1]-j
            flow[i,j,1]=max_yx[0]-i
            del src_ft
            del cos_map
            torch.cuda.empty_cache()
    return flow

def backwarp(tenIn, tenFlow):
    backwarp_tenGrid = {}
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(start=-1.0, end=1.0, steps=tenFlow.shape[3], dtype=tenFlow.dtype, device=tenFlow.device).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(start=-1.0, end=1.0, steps=tenFlow.shape[2], dtype=tenFlow.dtype, device=tenFlow.device).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])
        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).cuda()

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenIn.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenIn.shape[2] - 1.0) / 2.0)], 1)
    return torch.nn.functional.grid_sample(input=tenIn, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)

def splat_flow(frame1, frame2, flow, time):
    # frame1: tensor[1,c,h,w] cuda()
    # frame2: tensor[1,c,h,w] cuda()
    # flow  : tensor[h,w,2] cuda()
    flow = flow.permute(2,0,1).unsqueeze(0).cuda()
    tenMetric = torch.nn.functional.l1_loss(input=frame1, target=backwarp(tenIn=frame2, tenFlow=flow), reduction='none').mean([1], True)
    out_soft = softsplat(tenIn=frame1, tenFlow=flow*time, tenMetric=(-10.0 * tenMetric).clip(-10.0, 10.0), strMode='soft')
    return out_soft

def splat_flowmax(frame1, frame2, flow, time):
    # frame1: tensor[1,c,h,w] cuda()
    # frame2: tensor[1,c,h,w] cuda()
    # flow  : tensor[h,w,2] cuda()
    flow = flow.permute(2,0,1).unsqueeze(0).cuda()
    tenMetric = torch.nn.functional.l1_loss(input=frame1, target=backwarp(tenIn=frame2, tenFlow=flow), reduction='none').mean([1], True)
    # out_soft, mask = softsplat(tenIn=frame1, tenFlow=flow*time, tenMetric=(0.3 - tenMetric).clip(0.001, 1.0), strMode='max')
    return ((0.3-tenMetric).clip(0.001, 1.0))*time

def predict_z0(model, args,sup_res_h, sup_res_w):
    with torch.no_grad():
        invert_code, pred_x0_list = model.invert(args.source_image,
                                args.prompt,
                                guidance_scale=args.guidance_scale,
                                num_inference_steps=args.n_inference_step,
                                num_actual_inference_steps=args.feature_inversion,
                                return_intermediates=True)
        init_code = deepcopy(invert_code)
        model.scheduler.set_timesteps(args.n_inference_step)
        t = model.scheduler.timesteps[args.n_inference_step - args.feature_inversion]
        text_emb = model.get_text_embeddings(args.prompt).detach()
        unet_output, all_return_features = model.forward_unet_features(init_code, t, encoder_hidden_states=text_emb.repeat(2,1,1),
                layer_idx=args.unet_feature_idx)
        F0 = all_return_features[args.unet_feature_idx[0]]

        invert_code, pred_x0_list = model.invert(args.source_image,
                                args.prompt,
                                guidance_scale=args.guidance_scale,
                                num_inference_steps=args.n_inference_step,
                                num_actual_inference_steps=args.n_actual_inference_step,
                                return_intermediates=True)
        init_code = deepcopy(invert_code)
        pred_code = deepcopy(pred_x0_list[args.n_actual_inference_step])
        
        src_mask = torch.ones(2, 1, init_code.shape[2], init_code.shape[3]).cuda()
        input_code = torch.cat([pred_code, src_mask], 1)
        flow1to2   = feature_flow(F0, forward = True, f_size=(sup_res_h, sup_res_w)) 
        flow2to1   = feature_flow(F0, forward = False, f_size=(sup_res_h, sup_res_w))
        pred_list = []
        # cv2.imwrite(os.path.join(args.save_dir,'pred_0.png'), cv2.cvtColor(model.latent2image(pred_code[:1]), cv2.COLOR_BGR2RGB))
        # cv2.imwrite(os.path.join(args.save_dir,'pred_%d.png' %args.Time), cv2.cvtColor(model.latent2image(pred_code[-1:]), cv2.COLOR_BGR2RGB))

        for i in range(1,args.Time): 
            time = (i)/args.Time
            out_soft12 = splat_flow(input_code[:1], input_code[1:], flow1to2, time)
            out_soft21 = splat_flow(input_code[1:], input_code[:1], flow2to1, 1-time)
            mask =  out_soft12[:,-1] * out_soft21[:,-1]
            out_soft = ((1. - time) * out_soft12[0,:-1] + time * out_soft21[0,:-1]) * mask + out_soft21[0,:-1] * torch.clamp((out_soft21[:,-1]-mask),min=0) + out_soft12[0,:-1] * torch.clamp((out_soft12[:,-1]-mask),min=0) + ((1. - time) * input_code[0,:-1] + time * input_code[1,:-1]) * (1-torch.clamp((out_soft12[:,-1] + out_soft21[:,-1]),max=1))
            pred_list.append(out_soft.unsqueeze(0))
            # cv2.imwrite(os.path.join(args.save_dir,'pred_%d.png'  %i), cv2.cvtColor(model.latent2image(out_soft.unsqueeze(0)), cv2.COLOR_BGR2RGB))
        pred_list = torch.cat(pred_list,dim=0)
        
        torch.save(pred_list, os.path.join(args.save_dir,"pred_list.pt"))
        return flow1to2, flow2to1