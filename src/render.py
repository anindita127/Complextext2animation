import torch
import torch.nn as nn
from dataUtils import *
from model.model import *
from data import *
from common.transforms3dbatch import *
from common.parallel import parallel
from utils.kit_visualization import render, render4
from renderUtils import *
from argsUtils import argparseNloop

import numpy as np
from tqdm import tqdm
import pdb
import os


def render_all(args, exp_num):
    dir_name = args.save_dir
    assert dir_name "Directory not given."
    path2data = args.path2data
    dataset = args.dataset
    feats_kind = args.feats_kind
    render_list_ = args.render_list

    if render_list_ is not None:
        with open(render_list_, 'r') as f:
            render_list = f.readlines()
            render_list = {filename.strip() for filename in render_list}
    else:
        render_list = None

    data = Data(path2data, dataset, lmksSubset=[
                'all'], desc=None, load_data=False)

    # Load Skeleton
    skel = torch.load(open('dataProcessing/skeleton.p', 'rb'))
    filenames = []
    descriptions = []
    outputs = []

    feats_kind_dict = {'axis-angle': 'axs',
                       'fke': 'fke',
                       'rifke': 'csv'}

    idx = 1
    for tup in os.walk(dir_name):
        for filename in tup[2]:
            # print(filename)
            if filename.split('.')[-1] == feats_kind_dict[feats_kind]:
                if render_list:  # only render the files in render list
                    if filename.split('_')[0] not in render_list:
                        continue
                output = Path(tup[0])/'videos'/filename
                if not clean_render:  # only render files which do not exist. Useful if rendering was interrupted/incomplete
                    if output.with_suffix('.mp4').exists():
                        continue
                outputs.append(output.with_suffix('.mp4').as_posix())
                descriptions.append(get_description(
                    data.df, filename, path2data, feats_kind))

                os.makedirs(output.parent, exist_ok=True)
                filename = Path(tup[0])/filename
                filenames.append(filename.as_posix())
    filenum = len(filenames)
    print('{} files'.format(filenum))
    parallelRender(filenames, descriptions, outputs, skel, feats_kind, data)


if __name__ == '__main__':

    argparseNloop(render_all)
