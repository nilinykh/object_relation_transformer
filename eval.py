from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os, sys
import pickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import argparse
import misc.utils as utils
import torch

'''
To run inference set --language_eval=0
To run evaluation metrics set --language_eval=1
To visualize a sample set --dump_images 1 --num_images 100
(It is recommended to keep num_images small, as this will save num_images on disk under vis/)

e.g.: python eval.py \
        --dump_images 1 \
        --num_images 100 \
        --model /efs/home/sherdade/experiments/captioning/relation_rewritten_with_relu/model.pth \
        --infos_path /efs/home/sherdade/experiments/captioning/relation_rewritten_with_relu/infos_fc_transformer_bu_adaptive-best.pkl \
        --image_root /mydisk/data/captioning_data/coco/ \
        --input_json /mydisk/data/captioning_data/cocotalk.json \
        --input_fc_dir /mydisk/data/captioning_data/cocobu_adaptive_fc \
        --input_att_dir /mydisk/data/captioning_data/cocobu_adaptive_att \
        --input_box_dir /mydisk/data/captioning_data/cocobu_adaptive_box \
        --input_rel_box_dir=/mydisk/data/captioning_data/cocobu_adaptive_box_relative/ \
        --input_label_h5 /mydisk/data/captioning_data/cocotalk_label.h5  \
        --language_eval 1
'''


# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='./log_relation_transformer_bu_rl/model-best.pth',
                help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='./log_relation_transformer_bu_rl/infos_relation_transformer_bu-best.pkl',
                help='path to infos to evaluate')
# Basic options
parser.add_argument('--batch_size', type=int, default=1,
                help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=-1,
                help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=1,
                help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=1,
                help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=1,
                help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=0,
                help='Write image paths along with predictions into vis json? (1=yes,0=no)')

# Sampling options
parser.add_argument('--sample_max', type=int, default=1,
                help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--max_ppl', type=int, default=0,
                help='beam search by max perplexity or max probability.')
parser.add_argument('--beam_size', type=int, default=1,
                help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--group_size', type=int, default=1,
                help='used for diverse beam search. if group_size is 1, then it\'s normal beam search')
parser.add_argument('--diversity_lambda', type=float, default=0.5,
                help='used for diverse beam search. Usually from 0.2 to 0.8. Higher value of lambda produces a more diverse list')
parser.add_argument('--temperature', type=float, default=1.0,
                help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
parser.add_argument('--decoding_constraint', type=int, default=0,
                help='If 1, not allowing same word in a row')
# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default='',
                help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='/scratch/nikolai/',
                help='In case the image paths have to be preprended with a root path to an image folder')
# For evaluation on MSCOCO images from some split:
#parser.add_argument('--input_fc_dir', type=str, default='',
#                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_feat_dir', type=str, default='/scratch/nikolai/frcnn_train2014/',
                help='path to the h5file containing the preprocessed dataset')
#parser.add_argument('--input_box_dir', type=str, default='',
#                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_label_h5', type=str, default='./data/cocotalk_label.h5',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_json', type=str, default='./data/cocotalk.json',
                help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
parser.add_argument('--cnn_weight_dir', type=str, default='./data/imagenet_weights/',
                help='path to the directory containing the weights of a model trained on imagenet')
parser.add_argument('--split', type=str, default='train',
                help='if running on MSCOCO images, which split to use: val|test|train')
parser.add_argument('--coco_json', type=str, default='',
                help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')

parser.add_argument('--input_rel_box_dir',type=str, default='/scratch/nikolai/box_relative_train/',
                help="this directory contains the bboxes in relative coordinates for the corresponding image features in --input_att_dir")
# misc
parser.add_argument('--id', type=str, default='',
                help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
parser.add_argument('--verbose_beam', type=int, default=1,
                help='if we need to print out all beam search beams.')
parser.add_argument('--verbose_loss', type=int, default=0,
                help='if we need to calculate loss.')


opt = parser.parse_args()

# Load infos
with open(opt.infos_path, 'rb') as f:
    infos = pickle.load(f, encoding='latin1')

# override and collect parameters
#if len(opt.input_fc_dir) == 0:
#    opt.input_fc_dir = infos['opt'].input_fc_dir
#    opt.input_att_dir = infos['opt'].input_att_dir
#    opt.input_box_dir = infos['opt'].input_box_dir
#    opt.input_label_h5 = infos['opt'].input_label_h5
    
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
if len(opt.id) == 0:
    opt.id = infos['opt'].id
    
opt.seq_length = 16

ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval"]

for k in vars(infos['opt']).keys():
    if k not in ignore:
        if k in vars(opt):
            #assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
            # important for checking input_feat_dir variable
            if vars(opt)[k] != vars(infos['opt'])[k]:
                print(k, vars(opt)[k], vars(infos['opt'])[k])
                found_issue = k + ' option not consistent'
                if not utils.want_to_continue(found_issue):
                    exit()
        else:
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping

# Setup the model
model = models.setup(opt)
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()

crit = utils.LanguageModelCriterion()

# Create the Data Loader instance

if len(opt.image_folder) == 0:
  loader = DataLoader(opt)
else:
  loader = DataLoaderRaw({'folder_path': opt.image_folder,
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model,
                            'cnn_weight_dir': opt.cnn_weight_dir})

# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
# UNCOMMENT WHEN EVALUATING ON THE TEST SET, USE COCOTALK.JSON VOCAB WHEN RUNNING ON TRAIN SET
#loader.ix_to_word = infos['vocab']


# Set sample options
import eval_utils
loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader,
    vars(opt))

print('loss: ', loss)
if lang_stats:
  print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/vis.json', 'w'))
