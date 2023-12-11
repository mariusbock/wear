# ------------------------------------------------------------------------
# ActionFormer: Localizing Moments of Actions with Transformers
# ------------------------------------------------------------------------
# https://github.com/happyharrycn/actionformer_release
# ------------------------------------------------------------------------
# Adaption by: Marius Bock
# E-Mail: marius.bock@uni-siegen.de
# ------------------------------------------------------------------------
import os

import torch
import torch.nn as nn
import torch.utils.data
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from utils.data_utils import convert_segments_to_samples
from utils.os_utils import mkdir_if_missing

from .libs.modeling import make_meta_arch
from .libs.utils import (train_one_epoch, ANETdetection, save_checkpoint, make_optimizer, make_scheduler, ModelEma)
from .libs.datasets.datasets import make_data_loader, make_dataset
from .libs.utils.train_utils import valid_one_epoch
from .libs.core.config import _update_config


def run_actionformer(val_sbjs, cfg, ckpt_folder, ckpt_freq, resume, rng_generator, run):
    cfg = _update_config(cfg)
    split_name = cfg['dataset']['json_anno'].split('/')[-1].split('.')[0]
    mkdir_if_missing(os.path.join(ckpt_folder, 'ckpts'))
    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])
    
    train_dataset = make_dataset(cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset'])
    val_dataset = make_dataset(cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset'])
    
    # validation sensor data loading
    val_sens_data = None    
    for sbj in val_sbjs:
        sbj_data = pd.read_csv(os.path.join(cfg['dataset']['sens_folder'], sbj + '.csv'), index_col=False).replace({"label": cfg['label_dict']}).fillna(0)
        if val_sens_data is None:
            val_sens_data = sbj_data
        else:
            val_sens_data = pd.concat((val_sens_data, sbj_data))

    val_sens_data = val_sens_data.to_numpy()

    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']
    # data loaders
    train_loader = make_data_loader(train_dataset, True, rng_generator, **cfg['loader'])
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(val_dataset, False, None, 1, cfg['loader']['num_workers'])

    # model
    model = make_meta_arch(cfg['model']['model_name'], **cfg['model'])
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    print("Number of learnable parameters for ActionFormer: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    model_ema = ModelEma(model)

    # resume from a checkpoint?
    if resume:
        if os.path.isfile(resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(resume, map_location = lambda storage, loc: storage.cuda(cfg['devices'][0]))
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            return
    else:
        start_epoch = 0

    # start training
    max_epochs = cfg['opt'].get('early_stop_epochs', cfg['opt']['epochs'] + cfg['opt']['warmup_epochs'])
    t_losses, v_losses= np.array([]), np.array([])
    for epoch in range(start_epoch, start_epoch + max_epochs):
        # train for one epoch
        t_loss = train_one_epoch(train_loader, model, optimizer, scheduler, model_ema, cfg['train_cfg']['clip_grad_l2norm'])

        # save ckpt once in a while
        if (((ckpt_freq > 0) and ((epoch + 1) % ckpt_freq == 0))):
            save_states = { 
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_states['state_dict_ema'] = model_ema.module.state_dict()
            file_name = 'epoch_{:03d}_{}.pth.tar'.format(epoch + 1, split_name)
            save_checkpoint(save_states, False, file_folder=os.path.join(ckpt_folder, 'ckpts'), file_name=file_name)
        
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(val_dataset.json_anno, val_dataset.split[0], tiou_thresholds = val_db_vars['tiou_thresholds'])
        
        v_loss, v_segments = valid_one_epoch(val_loader, model)
        v_preds, v_gt, _ = convert_segments_to_samples(v_segments, val_sens_data, cfg['dataset']['sampling_rate'])
        
        if ((epoch + 1) == max_epochs):
            # save raw results (for later postprocessing)
            v_results = pd.DataFrame({
                'video-id' : v_segments['video-id'],
                't-start' : v_segments['t-start'].tolist(),
                't-end': v_segments['t-end'].tolist(),
                'label': v_segments['label'].tolist(),
                'score': v_segments['score'].tolist()
            })
            mkdir_if_missing(os.path.join(ckpt_folder, 'unprocessed_results'))
            np.save(os.path.join(ckpt_folder, 'unprocessed_results', 'v_preds_' + split_name), v_preds)
            np.save(os.path.join(ckpt_folder, 'unprocessed_results', 'v_gt_' + split_name), v_gt)
            v_results.to_csv(os.path.join(ckpt_folder, 'unprocessed_results', 'v_seg_' + split_name + '.csv'), index=False)

        val_mAP, _ = det_eval.evaluate(v_segments)
        conf_mat = confusion_matrix(v_gt, v_preds, normalize='true', labels=range(len(cfg['labels'])))
        v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
        v_prec = precision_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(cfg['labels'])))
        v_rec = recall_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(cfg['labels'])))
        v_f1 = f1_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(cfg['labels'])))

        # print to terminal
        block1 = 'Epoch: [{:03d}/{:03d}]'.format(epoch, max_epochs)
        block2 = 'TRAINING:\tavg. loss {:.2f}'.format(t_loss)
        block3 = 'VALIDATION:\tavg. loss {:.2f}'.format(v_loss)
        block4 = ''
        block4  += '\t\tAvg. mAP {:>4.2f} (%) '.format(np.nanmean(val_mAP) * 100)
        for tiou, tiou_mAP in zip(cfg['dataset']['tiou_thresholds'], val_mAP):
                block4 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(tiou_mAP*100)
        block5 = ''
        block5  += '\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
        block5  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
        block5  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
        block5  += ' F1 {:>4.2f} (%)'.format(np.nanmean(v_f1) * 100)
        
        print('\n'.join([block1, block2, block3, block4, block5]))
        t_losses = np.append(t_losses, t_loss)
        v_losses = np.append(v_losses, v_loss)

        if run is not None:
            run[split_name].append({"train_loss": t_loss, "val_loss": v_loss, "accuracy": np.nanmean(v_acc), "precision": np.nanmean(v_prec), "recall": np.nanmean(v_rec), 'f1': np.mean(v_f1), 'mAP': np.mean(val_mAP)}, step=epoch)
            for tiou, tiou_mAP in zip(cfg['dataset']['tiou_thresholds'], val_mAP):
                    run[split_name].append({'mAP@' + str(tiou): tiou_mAP}, step=epoch)

    return t_losses, v_losses, val_mAP, v_preds, v_gt 
