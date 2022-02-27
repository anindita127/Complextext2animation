import torch
import torch.nn as nn
import pickle
from dataUtils import *
from model.model_hierarchical_twostream import *
from model.model import *
from data import *
from common.transforms3dbatch import *
from torch._C import *

# from pycasper.name import Name
# from pycasper.BookKeeper import *
from BookKeeper import *
from argsUtils import argparseNloop
from renderUtils import parallelRender
from operator import add
import numpy as np
from tqdm import tqdm
import pdb
import math


def sample(args, exp_num, data=None):
    assert args.load, 'Model name not provided'
    assert os.path.isfile(args.load), 'Model file not found'

    args_subset = ['exp', 'cpk', 'model', 'time']
    book = BookKeeper(args, args_subset, args_dict_update={})
    args = book.args
    dir_name = book.name.dir(args.save_dir)
    # Training parameters
    path2data = args.path2data
    dataset = args.dataset
    lmksSubset = args.lmksSubset
    desc = args.desc
    split = (args.train_frac, args.dev_frac)
    idx_dependent = args.idx_dependent

    # hardcoded for sampling
    batch_size = args.batch_size
    time = 64
    chunks = args.chunks
    offset = args.offset
    # mask for delta
    mask = args.mask

    global feats_kind
    feats_kind = args.feats_kind
    s2v = args.s2v
    f_new = args.f_new
    curriculum = args.curriculum

    # Load data iterables
    if data is None:
        data = Data(path2data, dataset, lmksSubset, desc,
                    split, batch_size=batch_size,
                    time=time,
                    chunks=chunks,
                    offset=offset,
                    shuffle=False,
                    mask=mask,
                    feats_kind=feats_kind,
                    s2v=s2v,
                    f_new=f_new)

        print('Data Loaded')
    else:
        print('Data already loaded!! Yessss!')

    train = data.train.dataset.datasets
    dev = data.dev.dataset.datasets
    test = data.test.dataset.datasets

    # Create a model
    device = torch.device('cuda:{}'.format(args.cuda)
                          ) if args.cuda >= 0 else torch.device('cpu')
    input_shape = data.input_shape
    modelKwargs = {}
    modelKwargs.update(input_shape)
    modelKwargs.update(args.modelKwargs)

    # getting the input_size
    if args.s2v:
        input_size = 300
    elif args.desc:
        input_size = len(args.desc)
    else:
        input_size = 0

    model = PoseGenerator(chunks, input_size,
                          Seq2SeqKwargs=modelKwargs, load=None)
    model.to(device).double()

    print('Model Created')

    # Load model
    if args.load:
        print('Loading Model')
        book._load_model(model)

    # Loss function
    mse_loss = nn.MSELoss(reduction='mean')

    # Transforms
    global columns
    columns = get_columns(feats_kind, data)
    pre = Transforms(args.transforms, columns, args.seed,
                     mask, feats_kind, dataset, f_new)

    def loop(model, data, dataLoaders, pre, batch_size, desc='train'):
        sentences = {}
        total_loss_cee = 0
        total_loss_see = 0
        model.eval()

        Tqdm = tqdm(dataLoaders, desc=desc +
                    ' {:.4f}'.format(0), leave=False, ncols=20)
        for count, loader in enumerate(Tqdm):
            loader = DataLoader(loader, batch_size=1, shuffle=False)
            outputs_list = []
            start_trajectory_list = []
            for kount, batch in enumerate(loader):
                model.zero_grad()

                X, Y, s2v, path = batch['input'], batch['output'], batch['desc'], batch['path']
                pose, trajectory, start_trajectory = X
                pose_gt, trajectory_gt, start_trajectory_gt = Y

                x = torch.cat((trajectory, pose), dim=-1)
                y = torch.cat((trajectory_gt, pose_gt), dim=-1)

                x = x.to(device)
                y = y.to(device)
                start_trajectory_gt = start_trajectory_gt.to(device)
                if isinstance(s2v, torch.Tensor):
                    s2v = s2v.to(device)

                # Transform before the model
                x = pre.transform(x)
                y = pre.transform(y)
                z, lz, gp, gs = model.sample_encoder(s2v, x)
                total_loss_cee += math.sqrt(mse_loss(z, lz))
                total_loss_see += math.sqrt(mse_loss(gp, gs))
                # update tqdm
                # Tqdm.set_description(desc+' {:.4f} {:.4f}'.format(running_loss/running_count, running_internal_loss/running_count))
                Tqdm.refresh()

                x = x.detach()
                y = y.detach()
                # loss = loss.detach()

            if count >= 0 and args.debug:  # debugging by overfitting
                break

        return total_loss_cee, total_loss_see

    # Sample
    ndev = len(dev)
    ntrain = len(train)
    ntest = len(test)
    dev_loss_cee, dev_loss_see = loop(model, data, dev, pre, batch_size, 'dev')
    with open(dir_name+"/CEE_dev.bin", "wb") as fp:
        pickle.dump(dev_loss_cee / ndev, fp)
    with open(dir_name+"/SEE_dev.bin", "wb") as fp:
        pickle.dump(dev_loss_see / ndev, fp)

    test_loss_cee, test_loss_see = loop(
        model, data, test, pre, batch_size, 'test')
    with open(dir_name+"/CEE_test.bin", "wb") as fp:
        pickle.dump(test_loss_cee / ntest, fp)
    with open(dir_name+"/SEE_test.bin", "wb") as fp:
        pickle.dump(test_loss_see / ntest, fp)

    train_loss_cee, train_loss_see = loop(
        model, data, train, pre, batch_size, 'train')
    with open(dir_name+"/CEE_train.bin", "wb") as fp:
        pickle.dump(train_loss_cee / ntrain, fp)
    with open(dir_name+"/SEE_train.bin", "wb") as fp:
        pickle.dump(train_loss_see / ntrain, fp)


if __name__ == '__main__':
    argparseNloop(sample)
