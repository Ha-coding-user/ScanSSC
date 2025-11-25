import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

semantic_kitti_class_frequencies = np.array(
    [
        5.41773033e09,
        1.57835390e07,
        1.25136000e05,
        1.18809000e05,
        6.46799000e05,
        8.21951000e05,
        2.62978000e05,
        2.83696000e05,
        2.04750000e05,
        6.16887030e07,
        4.50296100e06,
        4.48836500e07,
        2.26992300e06,
        5.68402180e07,
        1.57196520e07,
        1.58442623e08,
        2.06162300e06,
        3.69705220e07,
        1.15198800e06,
        3.34146000e05,
    ]
)

kitti_class_names = [
    "empty",
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]


def KL_sep(p, target):
    """
    KL divergence on nonzeros classes
    """
    nonzeros = target != 0
    nonzero_p = p[nonzeros]
    kl_term = F.kl_div(torch.log(nonzero_p), target[nonzeros], reduction="sum")
    return kl_term


def geo_scal_loss(pred, ssc_target, ignore_index=255, non_empty_idx=0):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, non_empty_idx]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != ignore_index
    nonempty_target = ssc_target != non_empty_idx
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    eps = 1e-5
    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / (nonempty_probs.sum()+eps)
    recall = intersection / (nonempty_target.sum()+eps)
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / ((1 - nonempty_target).sum()+eps)
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )
    
def cum_loss(pred, ssc_target, ignore_index=255, non_empty_idx=0):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, non_empty_idx]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != ignore_index
    nonempty_target = ssc_target != non_empty_idx
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    eps = 1e-5
    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / (nonempty_probs.sum()+eps)
    recall = intersection / (nonempty_target.sum()+eps)
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / ((1 - nonempty_target).sum()+eps)
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )


def sem_scal_loss(pred, ssc_target, ignore_index=255):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != ignore_index
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i]

        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy(
                    precision, torch.ones_like(precision)
                )
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.ones_like(specificity)
                )
                loss_class += loss_specificity
            loss += loss_class
    return loss / count


def CE_ssc_loss(pred, target, class_weights=None, ignore_index=255):
    """
    :param: prediction: the predicted tensor, must be [BS, C, ...]
    """
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=ignore_index, reduction="mean"
    )
    loss = criterion(pred, target.long())

    return loss

def vel_loss(pred, gt):
    return F.l1_loss(pred, gt)

def scan_loss(pred, target):
    pred_cumsum_height = torch.cumsum(pred, dim=4) 
    pred_cumsum_height /= (torch.arange(1,pred.size()[4]+1, device='cuda').reshape(1, 1, 1, 1, pred.size()[4]))
    
    pred_cumsum_depth = torch.cumsum(pred.flip((2)), dim=2) 
    pred_cumsum_depth /= (torch.arange(1,pred.size()[2]+1, device='cuda').reshape(1, 1, pred.size()[2], 1, 1))     
        
    pred_cumsum_width_left = torch.cumsum(pred[:,:,:,:pred.size(3)//2,:], dim=3)
    pred_cumsum_width_left /= (torch.arange(1,(pred.size(3)//2)+1, device='cuda').reshape(1, 1, 1, pred.size(3)//2, 1))
    pred_cumsum_width_right = torch.cumsum(pred[:,:,:,pred.size(3)//2:,:].flip((3)), dim=3) 
    pred_cumsum_width_right /= (torch.arange(1,(pred.size(3)//2)+1, device='cuda').reshape(1, 1, 1, pred.size(3)//2, 1))
    pred_cumsum_width = torch.cat([pred_cumsum_width_left, pred_cumsum_width_right], dim=3)
    
    
    target_occ = torch.where(target!=255, target, 20)
    target_cum_height = torch.cumsum(F.one_hot(target_occ.type(torch.long)).permute(0,4,1,2,3), dim=4).type(torch.float)
    target_cum_height /= (torch.arange(1,33, device='cuda').repeat(1, 256, 256, 1).type(torch.float)).unsqueeze(0)
    target_cum_height = target_cum_height[:, :20, :, :, :]
    
    target_cum_depth = torch.cumsum(F.one_hot(target_occ.type(torch.long)).permute(0,4,1,2,3).flip((2)), dim=2).type(torch.float)
    target_cum_depth /= (torch.arange(1,257, device='cuda').repeat(1, 256, 32, 1).permute(0,3,1,2).type(torch.float)).unsqueeze(0)
    target_cum_depth = target_cum_depth[:, :20, :, :, :]
    
    target_cum_width_left = torch.cumsum(F.one_hot(target_occ.type(torch.long)[:,:,:target_occ.size(2)//2,:]).permute(0,4,1,2,3), dim=3).type(torch.float) 
    target_cum_width_left /= (torch.arange(1,129, device='cuda').repeat(1, 256, 32, 1).permute(0,1,3,2).type(torch.float)).unsqueeze(0)
    target_cum_width_right = torch.cumsum(F.one_hot(target_occ.type(torch.long)[:,:,target_occ.size(2)//2:,:]).permute(0,4,1,2,3).flip((3)), dim=3).type(torch.float)
    target_cum_width_right /= (torch.arange(1,129, device='cuda').repeat(1, 256, 32, 1).permute(0,1,3,2).type(torch.float)).unsqueeze(0)
    target_cum_width = torch.cat([target_cum_width_left, target_cum_width_right], dim=3)[:, :20, :, :, :]
    
    loss_height = F.cross_entropy(pred_cumsum_height.float(),target_cum_height, reduction='mean')
    loss_depth = F.cross_entropy(pred_cumsum_depth.float(),target_cum_depth, reduction='mean')
    loss_width = F.cross_entropy(pred_cumsum_width.float(),target_cum_width, reduction='mean')
    
    return loss_height + loss_depth + loss_width
