from tqdm import tqdm
import math
import torch
import torch.nn.functional as F
import torch.nn as nn

import clip
import gc

LOWEST = -1
HIGHEST = 1
EPSILON = 0.01
Z_EPSILON = 1e-7
LOGIT_BETA = 4
RELEVANCE_RECT = -1e-6



def FDT(ft, sd, ft_dim, sd_dim, q_map):
    temperature = 2
    # q_map = nn.Sequential(
    #         nn.LayerNorm(ft_dim),
    #         nn.Linear(ft_dim, sd_dim),
    #         nn.GELU(),
    #         nn.LayerNorm(sd_dim),
    #         nn.Linear(sd_dim, sd_dim)
    #     ).to(ft.dtype).cuda()

    q = q_map(ft).to(torch.float16)
    # # ft = nn.LayerNorm(1000, device='cuda')(ft).to(torch.float16)
    # q = ft.to(torch.float16)
    
    k = sd.transpose(0, 1) 

    inner_dot = torch.matmul(q, k) 
    inner_dot = inner_dot / math.sqrt(sd_dim)
    inner_dot = inner_dot / temperature

    att_activation = nn.Softmax(dim=-1)
    att_weight = att_activation(inner_dot) 

    att_ft = att_weight @ sd  

    # aa = nn.Parameter(torch.tensor(0.001)).to(torch.float16)
    # att_ft = aa*att_weight + ft
     
    # att_weight = torch.matmul(q, k)
    # aa = nn.Parameter(torch.tensor(0.001)).to(torch.float16)
    att_ft = att_ft + ft
    
    return att_ft


def accuracy(output, label, n, topk=(1, 5)):
    # pred = output.topk(max(topk), 1, True, True)[1].t()
    output_in = output.topk(1, dim=1, largest=True).indices

    output_c = torch.zeros_like(output)
    output_c.scatter_(1, output_in, 1)

    correct = output_c.eq(label)
    # 遍历张量的每一行，检查是否有 False，并将结果存储在向量中
    result_vector = (~correct).sum(-1).bool().int()
    
    re = (result_vector == 0).sum().item() / n

    return 100 * re



def fenlei_model(clip_adapter, dino_adapter, beta, alpha, clip_cache_values,
                 dino_cache_values, clip_image_features, dino_image_features, clip_weights, sd, sd_dim, a1, a2, a3, q_map):

    # learnable FDT
    # space_dict = nn.Parameter(torch.randn(sd_num, sd_dim))


    clip_affinity = clip_adapter(clip_image_features)
    clip_cache_logits = ((-1) * (beta - beta * clip_affinity)).exp() @ clip_cache_values
    # clip_cache_logits = nn.LayerNorm(1000, device='cuda')(clip_cache_logits.float())
    dino_affinity = dino_adapter(dino_image_features).to(dino_cache_values.dtype)
    dino_cache_logits = ((-1) * (beta - beta * dino_affinity)).exp() @ dino_cache_values
    # dino_cache_logits = nn.LayerNorm(1000, device='cuda')(dino_cache_logits.float())
    clip_logits = 100. * clip_image_features @ clip_weights
    # clip_logits = nn.LayerNorm(1000, device='cuda')(clip_logits.float())

    # a1 = nn.Parameter(torch.tensor(0.001)).to(torch.float16)
    # a2 = nn.Parameter(torch.tensor(0.001)).to(torch.float16)
    # a3 = nn.Parameter(torch.tensor(0.001)).to(torch.float16)
    clip_logits = FDT(clip_logits, sd, clip_logits.size()[1], sd_dim, q_map)
    clip_cache_logits = FDT(clip_cache_logits, sd, clip_cache_logits.size()[1], sd_dim, q_map)
    dino_cache_logits = FDT(dino_cache_logits, sd, dino_cache_logits.size()[1], sd_dim, q_map)

    cache_logits = logits_fuse(clip_logits, [clip_cache_logits, dino_cache_logits])
    tip_logits = clip_logits + cache_logits * alpha
    # tip_logits = nn.LayerNorm(1000, device='cuda')(tip_logits.float())

    # print("c_value:", clip_cache_values.size(), "d_value", dino_cache_values.size())
    # print("clip_image_features:", clip_image_features, "size", clip_image_features.size())
    # print("dino_image_features:", dino_image_features, "size", dino_image_features.size())
    # print("clip_affinity:", clip_affinity, "size:", clip_affinity.size())
    # print("clip_cache_logits:", clip_cache_logits, "size:", clip_cache_logits.size())
    # print("dino_affinity:", dino_affinity, "size:", dino_affinity.size())
    # print("dino_cache_logits:", dino_cache_logits, "size:", dino_cache_logits.size())
    # print("clip_logits:", clip_logits, "size:", clip_logits.size())
    # print("cache_logits:", cache_logits, "size:", cache_logits.size())
    # print("tip_logits", tip_logits, "size:", tip_logits.size())

    return tip_logits


def poly_loss(
        x,
        target,
        eps: float = 2.0,
        weight = None,
        ignore_index: int = -1,
        reduction: str = "mean",
):
    """Implements the Poly1 loss from `"PolyLoss: A Polynomial Expansion Perspective of Classification Loss
    Functions" <https://arxiv.org/pdf/2204.12511.pdf>`_.
    Args:
        x (torch.Tensor[N, K, ...]): predicted probability
        target (torch.Tensor[N, K, ...]): target probability
        eps (float, optional): epsilon 1 from the paper
        weight (torch.Tensor[K], optional): manual rescaling of each class
        ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
        reduction (str, optional): reduction method
    Returns:
        torch.Tensor: loss reduced with `reduction` method
    """

    # log(P[class]) = log_softmax(score)[class]
    logpt = F.log_softmax(x, dim=-1)
    logpt = F.log_softmax(logpt, dim=-1)

    # Compute pt and logpt only for target classes (the remaining will have a 0 coefficient)
    logpt = logpt.transpose(1, 0).flatten(1).gather(0, target.view(1, -1)).squeeze()
    # Ignore index (set loss contribution to 0)
    valid_idxs = torch.ones(target.view(-1).shape[0], dtype=torch.bool, device=x.device)
    if ignore_index >= 0 and ignore_index < x.shape[1]:
        valid_idxs[target.view(-1) == ignore_index] = False

    # Get P(class)
    loss = -1 * logpt + eps * (1 - logpt.exp())

    # Weight
    if weight is not None:
        # Tensor type
        if weight.type() != x.data.type():
            weight = weight.type_as(x.data)
        logpt = weight.gather(0, target.data.view(-1)) * logpt

    # Loss reduction
    if reduction == "sum":
        loss = loss[valid_idxs].sum()
    elif reduction == "mean":
        loss = loss[valid_idxs].mean()
    else:
        # if no reduction, reshape tensor like target
        loss = loss.view(*target.shape)

    return loss


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def gpt_clip_classifier(classnames, gpt_prompts, clip_model, template):
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = []
            for t in gpt_prompts[classname]:
                texts.append(t)
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        print("class_embeddings.size:", class_embeddings.size())
        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights

def build_clip_cache_model(cfg, clip_model, train_loader_cache):

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []
        cache_s_keys = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []
                train_s_features = []
                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    g_image_features = image_features[0]
                    s_image_features = image_features[1:]
                    train_features.append(g_image_features)
                    train_s_features.append(s_image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
                cache_s_keys.append(torch.cat(train_s_features, dim=1).unsqueeze(0))

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)

        cache_s_keys = torch.cat(cache_s_keys, dim=0).mean(dim=0)
        cache_s_keys /= cache_s_keys.norm(dim=-1, keepdim=True)
        cache_s_keys = cache_s_keys.permute(1, 0, 2)

        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()
    
        torch.save(cache_keys, cfg['cache_dir'] + '/clip_keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/clip_values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/clip_keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/clip_values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values, cache_s_keys
        # 1024/16000   16000/1000    16000/49/1024
def build_dino_cache_model(cfg, dino_model, train_loader_cache):
    
    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []
        print("dino")
        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = dino_model(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                # print("image_features.size:", image_features.size())

                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
        # print("train_features.len:", len(train_features), "cache_keys.len:", len(cache_keys))

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        # print("cache_keys.size:", cache_keys.size())
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        # print("target.size:", target.size())
        # print("cache_values.size:", cache_values.size())

        torch.save(cache_keys, cfg['cache_dir'] + '/dino_keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/dino_values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/dino_keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/dino_values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values

def build_clip_dalle_cache_model(cfg, clip_model, train_loader_cache):
    
    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []
        cache_s_keys = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []
                train_s_features = []
                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    g_image_features = image_features[0]
                    s_image_features = image_features[1:]
                    train_features.append(g_image_features)
                    train_s_features.append(s_image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
                cache_s_keys.append(torch.cat(train_s_features, dim=1).unsqueeze(0))

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)

        cache_s_keys = torch.cat(cache_s_keys, dim=0).mean(dim=0)
        cache_s_keys /= cache_s_keys.norm(dim=-1, keepdim=True)
        cache_s_keys = cache_s_keys.permute(1, 0, 2)

        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/clip_dalle_keys_' + str(cfg['dalle_shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/clip_dalle_values_' + str(cfg['dalle_shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/clip_dalle_keys_' + str(cfg['dalle_shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/clip_dalle_values_' + str(cfg['dalle_shots']) + "shots.pt")

    return cache_keys, cache_values, cache_s_keys

def build_dino_dalle_cache_model(cfg, dino_model, train_loader_cache):
    
    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = dino_model(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/dino_dalle_keys_' + str(cfg['dalle_shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/dino_dalle_values_' + str(cfg['dalle_shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/dino_dalle_keys_' + str(cfg['dalle_shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/dino_dalle_values_' + str(cfg['dalle_shots']) + "shots.pt")

    return cache_keys, cache_values


def pre_CLIP_load_features(cfg, split, clip_model, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features = image_features[0]
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_clip_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_clip_l.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_clip_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_clip_l.pt")
    
    return features, labels


def pre_DINO_load_features(cfg, split, dino_model, loader):
    
    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = dino_model(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_dino_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_dino_l.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_dino_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_dino_l.pt")
    
    return features, labels


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha

def search_no_clip_hp(cfg, cache_keys, cache_values, features, labels, adapter=None):
    
    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features).to(torch.float16)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                # clip_logits = 100. * features @ clip_weights
                # tip_logits = clip_logits + cache_logits * alpha
                tip_logits = cache_logits
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha


def search_ensemble_hp(cfg, 
                    clip_cache_keys, 
                    clip_cache_values, 
                    clip_features, 
                    dino_cache_keys, 
                    dino_cache_values, 
                    dino_features, 
                    labels, 
                    clip_weights, 
                    clip_adapter=None, 
                    dino_adapter=None):
    
    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0
        # dj修改，加入下一行代码，目的是该段程序不计算参数梯度，修改缩进
        with torch.no_grad():
            for beta in beta_list:
                for alpha in alpha_list:
                    # dj修改，添加下面两行代码，目的：清内存
                    gc.collect()
                    torch.cuda.empty_cache()

                    if clip_adapter:
                        clip_affinity = clip_adapter(clip_features)
                        dino_affinity = dino_adapter(dino_features).to(dino_cache_values)
                    else:
                        clip_affinity = clip_features @ clip_cache_keys
                        dino_affinity = (dino_features @ dino_cache_keys).to(dino_cache_values)

                    clip_cache_logits = ((-1) * (beta - beta * clip_affinity)).exp() @ clip_cache_values
                    dino_cache_logits = ((-1) * (beta - beta * dino_affinity)).exp() @ dino_cache_values
                    clip_logits = 100. * clip_features @ clip_weights
                    cache_logits = logits_fuse(clip_logits, [clip_cache_logits, dino_cache_logits])
                    tip_logits = clip_logits + cache_logits * alpha
                    acc = cls_acc(tip_logits, labels)

                    if acc > best_acc:
                        print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                        best_acc = acc
                        best_beta = beta
                        best_alpha = alpha
        # dj修改以上
        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))
        with open("best.txt","w") as f:
            f.write("After searching, the best accuarcy: {:.2f}.\n".format(best_acc))
    return best_beta, best_alpha


# clip zero_shot as baseline
def logits_fuse(zero_logtis, logits, normalize='mean'):
    # normalize logits
    softmax_fun = nn.Softmax(dim=1)
    if normalize == 'softmax':
        zero_logtis = softmax_fun(zero_logtis)
    elif normalize =='linear':
        zero_logtis /= torch.norm(zero_logtis, p=2, dim=1, keepdim=True)
    elif normalize == 'mean':
        logits_std = torch.std(zero_logtis, dim=1, keepdim=True)
        logits_mean = torch.mean(zero_logtis, dim=1, keepdim=True)
        zero_logtis = (zero_logtis - logits_mean) / logits_std
    else:
        raise("error normalize!")
    similarity_matrix = []
    normalize_logits = []
    for logit in logits:
        if normalize == 'softmax':
            current_normalize_logits = softmax_fun(logit)
        elif normalize =='linear':
            current_normalize_logits = logit / torch.norm(logit, p=2, dim=1, keepdim=True)
        elif normalize == 'mean':
            logits_std = torch.std(logit, dim=1, keepdim=True)
            logits_mean = torch.mean(logit, dim=1, keepdim=True)
            current_normalize_logits = (logit - logits_mean) / logits_std
        else:
            raise("error normalize!")
        current_similarity = current_normalize_logits * zero_logtis
        current_similarity = torch.sum(current_similarity, dim=1, keepdim=True)
        similarity_matrix.append(current_similarity)
        normalize_logits.append(current_normalize_logits)

    # print("current_similarity.size:", current_similarity.size())
    # print("similarity_matrix.len", len(similarity_matrix))

    similarity_matrix = torch.stack(similarity_matrix, dim=-2)
    # print("similarity_matrix", similarity_matrix.size())
    similarity_matrix = softmax_fun(similarity_matrix)
    # print("similarity_matrix", similarity_matrix.size())

    normalize_logits = torch.stack(normalize_logits, dim=-2)
    result_logits = torch.sum(normalize_logits * similarity_matrix, dim=1)

    return result_logits
def logits_fuse_s(zero_logtis, logits, normalize='mean'):
    # normalize logits
    softmax_fun = nn.Softmax(dim=1)
    if normalize == 'softmax':
        zero_logtis = softmax_fun(zero_logtis)
    elif normalize =='linear':
        zero_logtis /= torch.norm(zero_logtis, p=2, dim=1, keepdim=True)
    elif normalize == 'mean':
        logits_std = torch.std(zero_logtis, dim=1, keepdim=True)
        logits_mean = torch.mean(zero_logtis, dim=1, keepdim=True)
        zero_logtis = (zero_logtis - logits_mean) / logits_std
    else:
        raise("error normalize!")
    similarity_matrix = []
    normalize_logits = []
    for logit in logits:
        if normalize == 'softmax':
            current_normalize_logits = softmax_fun(logit)
        elif normalize =='linear':
            current_normalize_logits = logit / torch.norm(logit, p=2, dim=1, keepdim=True)
        elif normalize == 'mean':
            logits_std = torch.std(logit, dim=1, keepdim=True)
            logits_mean = torch.mean(logit, dim=1, keepdim=True)
            current_normalize_logits = (logit - logits_mean) / logits_std
        else:
            raise("error normalize!")
        current_similarity = current_normalize_logits * zero_logtis
        current_similarity = torch.sum(current_similarity, dim=1, keepdim=True)
        similarity_matrix.append(current_similarity)
        normalize_logits.append(current_normalize_logits)
    similarity_matrix = torch.stack(similarity_matrix, dim=-2)
    similarity_matrix = softmax_fun(similarity_matrix)
    count = 0
    for i in similarity_matrix:
        if i[0]>0.4 and i[0]<0.6:
            count += 1
    normalize_logits = torch.stack(normalize_logits, dim=-2)
    result_logits = torch.sum(normalize_logits * similarity_matrix, dim=1)

    return result_logits, count
