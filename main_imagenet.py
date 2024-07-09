import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models as torchvision_models

from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
import dino.utils as utils
import itertools
import json
import gc
from launch import *



def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format', default="MFDFA-main/configs/imagenet/8shot.yaml")
    args = parser.parse_args()

    return args

# key是图片，values是文本
def cascaded_model(cfg, 
                            clip_cache_keys, 
                            clip_cache_values, 
                            clip_test_features, 
                            dino_cache_keys, 
                            dino_cache_values, 
                            dino_test_features, 
                            test_labels, 
                            clip_weights, 
                            clip_model, 
                            dino_model, 
                            train_loader_F,
                            dalle_train_loader_F):
    
    clip_adapter = nn.Linear(clip_cache_keys.shape[0], clip_cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    clip_adapter.weight = nn.Parameter(clip_cache_keys.t())
    dino_adapter = nn.Linear(dino_cache_keys.shape[0], dino_cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    dino_adapter.weight = nn.Parameter(dino_cache_keys.t())

    print("clip_cache_keys.szie:", clip_cache_keys.size(), clip_cache_keys.shape[0], clip_cache_keys.shape[1])
    print("dino_cache_keys.szie:", dino_cache_keys.size(), dino_cache_keys.shape[0], dino_cache_keys.shape[1])

    #FDT
    sd_num = cfg['sd_num']
    sd_dim = 1000
    sd = nn.Parameter(torch.eye(sd_num, device=torch.device("cuda"), dtype=clip_model.dtype))
    
    q_map = nn.Sequential(
            nn.LayerNorm(1000),
            nn.Linear(1000, sd_dim),
            nn.GELU(),
            nn.LayerNorm(sd_dim),
            nn.Linear(sd_dim, sd_dim)
        ).to(clip_model.dtype).cuda()

    a1 = nn.Parameter(torch.tensor(0.001, device=torch.device("cuda"), dtype=torch.float16))
    a2 = nn.Parameter(torch.tensor(0.001, device=torch.device("cuda"), dtype=torch.float16))
    a3 = nn.Parameter(torch.tensor(0.001, device=torch.device("cuda"), dtype=torch.float16))

    optimizer = torch.optim.AdamW(
        itertools.chain(dino_adapter.parameters(), clip_adapter.parameters(), q_map.parameters()),
        lr=cfg['lr'], 
        eps=1e-4)
    optimizer.add_param_group({"params": [a1, a2, a3, sd]})
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    
    for train_idx in range(cfg['train_epoch']):
        # Train
        is_t_e = 0
        clip_adapter.train()
        dino_adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []

        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        # origin image
        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                clip_image_features = clip_model.encode_image(images)
                clip_image_features = clip_image_features[0]
                clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)
                dino_image_features = dino_model(images)
                dino_image_features /= dino_image_features.norm(dim=-1, keepdim=True)

            tip_logits = fenlei_model(clip_adapter, dino_adapter, beta, alpha, clip_cache_values,
                                    dino_cache_values, clip_image_features, dino_image_features, clip_weights, sd, sd_dim, a1, a2, a3, q_map)

           
            
            loss1 = poly_loss(tip_logits, target)
            # loss1 = F.cross_entropy(tip_logits, target)
            loss = loss1

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # dalle image
        for i, (images, target) in enumerate(tqdm(dalle_train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                clip_image_features = clip_model.encode_image(images)
                clip_image_features = clip_image_features[0]
                clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)
                dino_image_features = dino_model(images)
                dino_image_features /= dino_image_features.norm(dim=-1, keepdim=True)

            tip_logits = fenlei_model(clip_adapter, dino_adapter, beta, alpha, clip_cache_values,
                                      dino_cache_values, clip_image_features, dino_image_features, clip_weights, sd, sd_dim, a1, a2, a3, q_map)

            
            
            loss1 = poly_loss(tip_logits, target)
            # loss1 = F.cross_entropy(tip_logits, target)
            loss = loss1

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        is_t_e = 1
        clip_adapter.eval()
        dino_adapter.eval()
        
        with torch.no_grad():

            tip_logits = fenlei_model(clip_adapter, dino_adapter, beta, alpha, clip_cache_values,
                                        dino_cache_values, clip_test_features, dino_test_features, clip_weights, sd, sd_dim, a1, a2, a3, q_map)
           
            print("test_labels.size:", test_labels.size())
            # acc = cls_acc(tip_logits_lrp, test_labels)
            acc = cls_acc(tip_logits, test_labels)

            print("**** MFDFA's test accuracy: {:.2f}. ****\n".format(acc))
            if acc > best_acc:
                best_acc = acc
                best_epoch = train_idx
                torch.save(clip_adapter.weight, cfg['cache_dir'] + "/best_F_clip_adapter_" + str(cfg['shots']) + "shots.pt")
                torch.save(dino_adapter.weight, cfg['cache_dir'] + "/best_F_dino_adapter_" + str(cfg['shots']) + "shots.pt")
       

    clip_adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_clip_adapter_" + str(cfg['shots']) + "shots.pt")
    dino_adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_dino_adapter_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, MFDFA's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")



def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir


    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['clip_backbone'])
    clip_model.eval()

    # DINO
    dino_model = torchvision_models.__dict__[cfg['dino_backbone']](num_classes=0)
    dino_model.fc = nn.Identity()
    dino_model.cuda()
    utils.load_pretrained_weights(dino_model, "path", "teacher", "vit_small'", 16)
    dino_model.eval()

    # ImageNet dataset
    random.seed(2)
    torch.manual_seed(1)

    print("Preparing ImageNet dataset.")
    imagenet = ImageNet(cfg['root_path'], cfg['shots'], preprocess)
    
    test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=32, num_workers=8, shuffle=False)

    train_loader_cache = torch.utils.data.DataLoader(imagenet.train, batch_size=128, num_workers=8, shuffle=False)
    train_loader_F = torch.utils.data.DataLoader(imagenet.train, batch_size=128, num_workers=8, shuffle=True)

    dalle_dataset = build_dataset(cfg['dalle_dataset'], cfg['root_path'], cfg['dalle_shots'])
    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    dalle_train_loader_cache = build_data_loader(data_source=dalle_dataset.train_x, batch_size=128, tfm=train_tranform, is_train=True, shuffle=False)
    dalle_train_loader_F = build_data_loader(data_source=dalle_dataset.train_x, batch_size=128, tfm=train_tranform, is_train=True, shuffle=True)
    
    with open(cfg['gpt3_prompt_file']) as f:
        gpt3_prompt = json.load(f)

    # Textual features
    print("Getting textual features as CLIP's classifier.")
    clip_weights = gpt_clip_classifier(imagenet.classnames, gpt3_prompt, clip_model, imagenet.template)
    

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    print("\nConstructing CLIP cache model.")
    clip_cache_keys, clip_cache_values, clip_cache_s_keys = build_clip_cache_model(cfg, clip_model, train_loader_cache)
    # 1024/16000     16000/1000         16000/49/1024
    print("\nConstructing DINO cache model.")
    dino_cache_keys, dino_cache_values = build_dino_cache_model(cfg, dino_model, train_loader_cache)

    print("\nConstructing cache model by dalle image.")
    print("\nConstructing CLIP cache model.")
    clip_dalle_cache_keys, clip_dalle_cache_values, clip_dalle_cache_s_keys = build_clip_dalle_cache_model(cfg, clip_model, dalle_train_loader_cache)
    print("\nConstructing DINO cache model.")
    dino_dalle_cache_keys, dino_dalle_cache_values = build_dino_dalle_cache_model(cfg, dino_model, dalle_train_loader_cache)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    print("\nLoading CLIP feature.")
    test_clip_features, test_labels = pre_CLIP_load_features(cfg, "test", clip_model, test_loader)
    print("\nLoading DINO feature.")
    test_dino_features, test_labels = pre_DINO_load_features(cfg, "test", dino_model, test_loader)

    clip_keys = torch.cat((clip_cache_keys, clip_dalle_cache_keys), dim=1)
    clip_values = torch.cat((clip_cache_values, clip_dalle_cache_values), dim=0)
    dino_keys = torch.cat((dino_cache_keys, dino_dalle_cache_keys), dim=1)
    dino_values = torch.cat((dino_cache_values, dino_dalle_cache_values), dim=0)


    # --------------------------------------Parameter-free Attention ------------------------------------


    with torch.no_grad():
        logits1 = []
        logits2 = []

        feat_t_as = torch.zeros((1024,1000)).to(clip_cache_keys.device)
        feat_t_att = []

        feat_v_as = []
        feat_v_att = []
        for i, feat_v in enumerate(tqdm(clip_s_keys)): # feat_v 1024/49
            A_weight = torch.matmul(feat_v.permute(1, 0), feat_t) * 2  # 49/1000
            A_weight1 = F.softmax(A_weight, dim=0)
            A_weight2 = F.softmax(A_weight, dim=1)

            feat_t_a = torch.matmul(feat_v, A_weight1) # 1024/1000
            feat_t_as = feat_t_as + feat_t_a
            
            feat_v_a = torch.matmul(A_weight2, feat_t.permute(1, 0)) # 49/1024
            feat_v_a = feat_v_a.mean(0) + feat_v_a.max(0)[0] # 1024
            feat_v_as.append(feat_v_a)
            
            # 乘积计算分类结果
            l1 = 100. * clip_g_keys[i] @ feat_t_a # 1000
            l2 = 100. * feat_v_a @ feat_t # 1000
            logits1.append(l1.unsqueeze(0)) 
            logits2.append(l2.unsqueeze(0)) 


        # 注意力特征
        feat_t_att = feat_t_as / clip_s_keys.shape[0] # 1024/1000
        feat_v_att = torch.stack(feat_v_as).squeeze() # 18000/1024
        feat_v_att = feat_v_att.permute(1, 0)


        beta2_list = [i * (cfg['beta2'] - 0.001) / 200 + 0.001 for i in range(200)]
        beta3_list = [i * (cfg['beta3'] - 0.001) / 200 + 0.001 for i in range(200)]
        best_acc = 0.
        best_beta2 = 0.
        best_beta3 = 0.
        

        for beta2 in beta2_list:
            for beta3 in beta3_list:
                # logits = 100. * clip_g_keys @ feat_t
                # logits = logits + logits1 * beta2 + logits2 * beta3

                feat_t = clip_weights + beta2*feat_t_att
                keys = clip_g_keys + beta3*feat_v_att.permute(1, 0)
                keys = keys.to(torch.float16)
                feat_t = feat_t.to(torch.float16)
                logits = 100. * keys @ feat_t 
                acc = accuracy(logits, clip_values, n=clip_g_keys.size(0))

                if acc > best_acc:
                    print('New best setting, beta1: {:.4f}; beta2: {:.4f}; beta3: {:.4f}; Acc: {:.2f}'.format(1, beta2, beta3, acc))
                    # worrect = logits.eq(clip_values)
                    # print('logits:', logits)
                    best_acc = acc
                    best_beta2 = beta2
                    best_beta3 = beta3


        clip_keys = clip_keys + best_beta2*feat_v_att
        clip_weights = clip_weights + best_beta3*feat_t_att
    
    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
   
    cascaded_model(cfg, 
                            clip_keys,
                            clip_values, 
                            test_clip_features, 
                            dino_keys, 
                            dino_values, 
                            test_dino_features, 
                            test_labels, 
                            clip_weights, 
                            clip_model, 
                            dino_model, 
                            train_loader_F,
                            dalle_train_loader_F)

if __name__ == '__main__':
    main()
    


