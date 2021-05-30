import torch
import torch.nn as nn
import pickle
from dataUtils import *
from model.model_hierarchical_twostream import *
from data import *
from common.transforms3dbatch import *
from torch._C import *
from BookKeeper import *
from argsUtils import argparseNloop
from renderUtils import parallelRender
from operator import add
import numpy as np
from tqdm import tqdm
import pdb
import math


def get_mask_indices(mask):
    indices = []
    for i, m in enumerate(mask):
        if m:
            indices.append(i)
    return indices

# get delta based on mask


def local2global(outputs_list, start_trajectory_list, input_shape, trajectory_size, mask):
    # use only the first start_trajectory
    outputs = torch.cat(outputs_list, dim=0)
    outputs = outputs.view(-1, input_shape)

    # get only the first time-step
    start_trajectory = start_trajectory_list[0][0]

    mask = np.array(mask)
    indices = get_mask_indices(mask)
    start_trajectory = start_trajectory[indices]
    for t in range(outputs.shape[0]):
        outputs[t, indices] += start_trajectory
        start_trajectory = outputs[t, indices]

    return outputs


def toEuler(M, joints, euler_columns):
    columns = ['root_tx', 'root_ty', 'root_tz'] + \
        ['{}_{}'.format(joint, suffix)
         for joint in joints for suffix in ['rx', 'ry', 'rz']]
    M = M.values
    quats = M[:, 3:].reshape(M.shape[0], -1, 4)
    quats = quats/((quats**2).sum(axis=-1, keepdims=True)
                   ** 0.5)  # normalize the quaternions
    euler = quat2eulerbatch(quats, axes='sxyz').reshape(
        quats.shape[0], int(quats.shape[1]*3)) * 180/np.pi
    euler = np.concatenate([M[:, :3], euler], axis=-1)
    df = pd.DataFrame(data=euler, columns=columns)
    return df[euler_columns]


def toFKE(M, data, filename):
    ''' Convert RIFKE to FKE '''
    output_columns = data.raw_data.output_columns('rifke')
    # M[output_columns].to_csv(filename.with_suffix('.rifke').as_posix()) ## save rifke as well
    M = data.raw_data.rifke2fke(M[output_columns].values)
    filename = str(filename)[:-9] + 'fke.csv'
    ''' Save FKE '''
    M = M.reshape(M.shape[0], -1)
    output_columns = data.raw_data.output_columns('fke')
    pd.DataFrame(data=M,
                 columns=output_columns).to_csv(filename)


def sample(args, exp_num, data=None):
    assert args.load, 'Model name not provided'
    assert os.path.isfile(args.load), 'Model file not found'

    args_subset = ['exp', 'cpk', 'model', 'time']
    book = BookKeeper(args, args_subset, args_dict_update={})
    args = book.args
    dir_name = book.name.dir(args.save_dir)

    # dir_name = args.load[:-2]
    # Training parameters
    path2data = args.path2data
    dataset = args.dataset
    lmksSubset = args.lmksSubset
    desc = args.desc
    split = (args.train_frac, args.dev_frac)
    idx_dependent = args.idx_dependent

    # hardcoded for sampling
    batch_size = args.batch_size
    time = args.time
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

    # Transforms
    global columns
    columns = get_columns(feats_kind, data)
    pre = Transforms(args.transforms, columns, args.seed,
                     mask, feats_kind, dataset, f_new)

    def loop(model, data, dataLoaders, pre, batch_size, desc='train'):
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

                if kount == 0:
                    if modelKwargs.get('start_zero'):
                        start = torch.zeros_like(x[:, 0, :])
                        start = torch.rand_like(x[:, 0, :])
                    else:
                        start = x[:, 0, :]
                else:
                    start = y_cap[:, -1, :]

                if offset == 0:
                    #y_cap, internal_losses = model(x, s2v, train=False)

                    y_cap, internal_losses = model.sample(
                        s2v, time_steps=50, start=start)
                else:
                    assert 0, 'offset = {}, it must be 0 for now'.format(
                        offset)

                input_shape = sum([data.input_shape[key]
                                   for key in data.input_shape])
                trajectory_size = data.input_shape['trajectory_size']

                outputs_list.append(pre.inv_transform(y_cap))
                start_trajectory_list.append(start_trajectory_gt)

                # update tqdm
                # Tqdm.set_description(desc+' {:.4f} {:.4f}'.format(running_loss/running_count, running_internal_loss/running_count))
                Tqdm.refresh()

                x = x.detach()
                y = y.detach()
                # loss = loss.detach()
                y_cap = y_cap.detach()

            if outputs_list:
                outputs = local2global(
                    outputs_list, start_trajectory_list, input_shape, trajectory_size, data.mask)
                new_size = list(outputs.shape)
                new_size[0] *= loader.dataset.f_ratio
                outputs = outputs.repeat(
                    1, loader.dataset.f_ratio).view(new_size)

                outputs = outputs.detach().cpu().numpy()

                # copy outputs in the dataframe format
                #mat_full_temp = pd.DataFrame(data=np.zeros((outputs.shape[0], len(columns))), columns=loader.dataset.mat_full.columns)
                mat_full_temp = pd.DataFrame(data=np.zeros(
                    (outputs.shape[0], len(columns))), columns=columns)

                # copy all joints
                mat_full_temp.loc[:, columns] = outputs

                if feats_kind == 'rifke':
                    mat_full_temp['root_tx'] = 0
                    mat_full_temp['root_tz'] = 0

                #description = Data.tokenize(data.df[data.df[feats_kind] == loader.dataset.path2csv]['descriptions'].item())

                filename = Path(dir_name)/Path(desc)/Path(
                    loader.dataset.path2csv).relative_to(path2data).with_suffix('.csv')
                #filename = Path(dir_name)/Path(desc)/Path(loader.dataset.path2csv).relative_to(path2data).with_name(Path(loader.dataset.path2csv).stem + '_{}'.format(description))
                os.makedirs(filename.parent, exist_ok=True)
                if feats_kind == 'quaternion' or feats_kind == 'axis-angle':
                    data.raw_data.mat2csv(
                        mat_full_temp.values, filename, columns)
                elif feats_kind == 'rifke':
                    output_columns = data.raw_data.output_columns('rifke')
                    mat_full_temp[output_columns].to_csv(
                        filename)  # save rifke as well

                    # toFKE(mat_full_temp,
                    #       data,
                    #       filename)

            if count >= 0 and args.debug:  # debugging by overfitting
                break

    # Sample
    loop(model, data, train, pre, batch_size, 'train')
    loop(model, data, dev, pre, batch_size, 'dev')
    loop(model, data, test, pre, batch_size, 'test')


if __name__ == '__main__':
    argparseNloop(sample)
