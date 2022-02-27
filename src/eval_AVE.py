import torch
import torch.nn as nn
import pickle
from dataUtils import *
from model.model_hierarchical_twostream import *
from data import *
from common.transforms3dbatch import *
from torch._C import *

# from pycasper.name import Name
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

    ''' Save FKE '''
    M = M.reshape(M.shape[0], -1)
    output_columns = data.raw_data.output_columns('fke')
    pd.DataFrame(data=M,
                 columns=output_columns).to_csv(filename.as_posix())


def sample(args, exp_num, data=None):
    assert args.load, 'Model name not provided'
    assert os.path.isfile(args.load), 'Model file not found'

    args_subset = ['exp', 'cpk', 'model', 'time', 'chunks']
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
        running_loss = 0
        running_internal_loss = 0
        running_count = 0
        mean_pose_loss_ = 0
        trajectory_loss_ = 0
        rightfoot_loss_ = 0
        leftfoot_loss_ = 0
        rightankle_loss_ = 0
        leftankle_loss_ = 0
        rightknee_loss_ = 0
        leftknee_loss_ = 0
        righthip_loss_ = 0
        lefthip_loss_ = 0
        rightwrist_loss_ = 0
        leftwrist_loss_ = 0
        rightelbow_loss_ = 0
        leftelbow_loss_ = 0
        rightshoulder_loss_ = 0
        leftshoulder_loss_ = 0
        lmrot_loss_ = 0
        rmrot_loss_ = 0
        root_loss_ = 0
        upperneck_loss_ = 0
        lowerneck_loss_ = 0
        pelvis_loss_ = 0
        torso_loss_ = 0

        running_joint_count = 0
        running_mean_pose_count = 0
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
                    y_cap, internal_losses = model.sample(
                        s2v, time_steps=x.shape[-2], start=start)
                else:
                    assert 0, 'offset = {}, it must be 0 for now'.format(
                        offset)
                # y_cap = pre.inv_transform(y_cap)
                # y = pre.inv_transform(y)

                '''For GT'''
                trajectory_mean = torch.mean(y[..., 0])
                # y = y[...,1:-4].view(y[0],y[1],-1,3)
                pelvis_mean = torch.mean(
                    y[..., 1]) + torch.mean(y[..., 2]) + torch.mean(y[..., 3])
                torso_mean = torch.mean(
                    y[..., 4]) + torch.mean(y[..., 5]) + torch.mean(y[..., 6])
                lowerneck_mean = torch.mean(
                    y[..., 7]) + torch.mean(y[..., 8]) + torch.mean(y[..., 9])
                upperneck_mean = torch.mean(
                    y[..., 10]) + torch.mean(y[..., 11]) + torch.mean(y[..., 12])
                leftshoulder_mean = torch.mean(
                    y[..., 13]) + torch.mean(y[..., 14]) + torch.mean(y[..., 15])
                leftelbow_mean = torch.mean(
                    y[..., 16]) + torch.mean(y[..., 17]) + torch.mean(y[..., 18])
                leftwrist_mean = torch.mean(
                    y[..., 19]) + torch.mean(y[..., 20]) + torch.mean(y[..., 21])
                rightshoulder_mean = torch.mean(
                    y[..., 22]) + torch.mean(y[..., 23]) + torch.mean(y[..., 24])
                rightelbow_mean = torch.mean(
                    y[..., 27]) + torch.mean(y[..., 26]) + torch.mean(y[..., 25])
                rightwrist_mean = torch.mean(
                    y[..., 28]) + torch.mean(y[..., 29]) + torch.mean(y[..., 30])
                lefthip_mean = torch.mean(
                    y[..., 33]) + torch.mean(y[..., 32]) + torch.mean(y[..., 31])
                leftknee_mean = torch.mean(
                    y[..., 34]) + torch.mean(y[..., 35]) + torch.mean(y[..., 36])
                leftankle_mean = torch.mean(
                    y[..., 39]) + torch.mean(y[..., 38]) + torch.mean(y[..., 37])
                lmrot_mean = torch.mean(
                    y[..., 40]) + torch.mean(y[..., 41]) + torch.mean(y[..., 42])
                leftfoot_mean = torch.mean(
                    y[..., 45]) + torch.mean(y[..., 44]) + torch.mean(y[..., 43])
                righthip_mean = torch.mean(
                    y[..., 46]) + torch.mean(y[..., 47]) + torch.mean(y[..., 48])
                rightknee_mean = torch.mean(
                    y[..., 51]) + torch.mean(y[..., 50]) + torch.mean(y[..., 49])
                rightankle_mean = torch.mean(
                    y[..., 52]) + torch.mean(y[..., 53]) + torch.mean(y[..., 54])
                rmrot_mean = torch.mean(
                    y[..., 57]) + torch.mean(y[..., 56]) + torch.mean(y[..., 55])
                rightfoot_mean = torch.mean(
                    y[..., 58]) + torch.mean(y[..., 59]) + torch.mean(y[..., 60])
                root_mean = torch.mean(
                    y[..., 63]) + torch.mean(y[..., 62]) + torch.mean(y[..., 61])

                trajectory_var = torch.sum(
                    (y[..., 0] - trajectory_mean)**2) / (y.shape[1]-1)
                rightfoot_var = torch.sum(
                    (y[..., 58] + y[..., 59] + y[..., 60] - rightfoot_mean)**2) / (y.shape[1]-1)
                leftfoot_var = torch.sum(
                    (y[..., 43] + y[..., 44] + y[..., 45] - leftfoot_mean)**2) / (y.shape[1]-1)
                rightankle_var = torch.sum(
                    (y[..., 52] + y[..., 53] + y[..., 54] - rightankle_mean)**2) / (y.shape[1]-1)
                leftankle_var = torch.sum(
                    (y[..., 37] + y[..., 38] + y[..., 39] - leftankle_mean)**2) / (y.shape[1]-1)
                rightknee_var = torch.sum(
                    (y[..., 49] + y[..., 50] + y[..., 51] - rightknee_mean)**2) / (y.shape[1]-1)
                leftknee_var = torch.sum(
                    (y[..., 34] + y[..., 35] + y[..., 36] - leftknee_mean)**2) / (y.shape[1]-1)
                righthip_var = torch.sum(
                    (y[..., 46] + y[..., 47] + y[..., 48] - righthip_mean)**2) / (y.shape[1]-1)
                lefthip_var = torch.sum(
                    (y[..., 31] + y[..., 32] + y[..., 33] - lefthip_mean)**2) / (y.shape[1]-1)
                rightwrist_var = torch.sum(
                    (y[..., 28] + y[..., 29] + y[..., 30] - rightwrist_mean)**2) / (y.shape[1]-1)
                leftwrist_var = torch.sum(
                    (y[..., 19] + y[..., 20] + y[..., 21] - leftwrist_mean)**2) / (y.shape[1]-1)
                rightelbow_var = torch.sum(
                    (y[..., 25] + y[..., 26] + y[..., 27] - rightelbow_mean)**2) / (y.shape[1]-1)
                leftelbow_var = torch.sum(
                    (y[..., 16] + y[..., 17] + y[..., 18] - leftelbow_mean)**2) / (y.shape[1]-1)
                rightshoulder_var = torch.sum(
                    (y[..., 22] + y[..., 23] + y[..., 24] - rightshoulder_mean)**2) / (y.shape[1]-1)
                leftshoulder_var = torch.sum(
                    (y[..., 13] + y[..., 14] + y[..., 15] - leftshoulder_mean)**2) / (y.shape[1]-1)
                lmrot_var = torch.sum(
                    (y[..., 40] + y[..., 41] + y[..., 42] - lmrot_mean)**2) / (y.shape[1]-1)
                rmrot_var = torch.sum(
                    (y[..., 55] + y[..., 56] + y[..., 57] - rmrot_mean)**2) / (y.shape[1]-1)

                root_var = torch.sum(
                    (y[..., 61] + y[..., 62] + y[..., 63] - root_mean)**2) / (y.shape[1]-1)
                upperneck_var = torch.sum(
                    (y[..., 12] + y[..., 11] + y[..., 10] - upperneck_mean)**2) / (y.shape[1]-1)
                lowerneck_var = torch.sum(
                    (y[..., 7] + y[..., 8] + y[..., 9] - lowerneck_mean)**2) / (y.shape[1]-1)
                pelvis_var = torch.sum(
                    (y[..., 1] + y[..., 2] + y[..., 3] - pelvis_mean)**2) / (y.shape[1]-1)
                torso_var = torch.sum(
                    (y[..., 4] + y[..., 5] + y[..., 6] - torso_mean)**2) / (y.shape[1]-1)

                '''For generated'''
                jtrajectory_mean = torch.mean(y_cap[..., 0])
                # y_cap = y_cap[...,1:-4].view(y_cap[0],y_cap[1],-1,3)
                jpelvis_mean = torch.mean(
                    y_cap[..., 1]) + torch.mean(y_cap[..., 2]) + torch.mean(y_cap[..., 3])
                jtorso_mean = torch.mean(
                    y_cap[..., 4]) + torch.mean(y_cap[..., 5]) + torch.mean(y_cap[..., 6])
                jlowerneck_mean = torch.mean(
                    y_cap[..., 7]) + torch.mean(y_cap[..., 8]) + torch.mean(y_cap[..., 9])
                jupperneck_mean = torch.mean(
                    y_cap[..., 10]) + torch.mean(y_cap[..., 11]) + torch.mean(y_cap[..., 12])
                jleftshoulder_mean = torch.mean(
                    y_cap[..., 13]) + torch.mean(y_cap[..., 14]) + torch.mean(y_cap[..., 15])
                jleftelbow_mean = torch.mean(
                    y_cap[..., 16]) + torch.mean(y_cap[..., 17]) + torch.mean(y_cap[..., 18])
                jleftwrist_mean = torch.mean(
                    y_cap[..., 19]) + torch.mean(y_cap[..., 20]) + torch.mean(y_cap[..., 21])
                jrightshoulder_mean = torch.mean(
                    y_cap[..., 22]) + torch.mean(y_cap[..., 23]) + torch.mean(y_cap[..., 24])
                jrightelbow_mean = torch.mean(
                    y_cap[..., 27]) + torch.mean(y_cap[..., 26]) + torch.mean(y_cap[..., 25])
                jrightwrist_mean = torch.mean(
                    y_cap[..., 28]) + torch.mean(y_cap[..., 29]) + torch.mean(y_cap[..., 30])
                jlefthip_mean = torch.mean(
                    y_cap[..., 33]) + torch.mean(y_cap[..., 32]) + torch.mean(y_cap[..., 31])
                jleftknee_mean = torch.mean(
                    y_cap[..., 34]) + torch.mean(y_cap[..., 35]) + torch.mean(y_cap[..., 36])
                jleftankle_mean = torch.mean(
                    y_cap[..., 39]) + torch.mean(y_cap[..., 38]) + torch.mean(y_cap[..., 37])
                jlmrot_mean = torch.mean(
                    y_cap[..., 40]) + torch.mean(y_cap[..., 41]) + torch.mean(y_cap[..., 42])
                jleftfoot_mean = torch.mean(
                    y_cap[..., 45]) + torch.mean(y_cap[..., 44]) + torch.mean(y_cap[..., 43])
                jrighthip_mean = torch.mean(
                    y_cap[..., 46]) + torch.mean(y_cap[..., 47]) + torch.mean(y_cap[..., 48])
                jrightknee_mean = torch.mean(
                    y_cap[..., 51]) + torch.mean(y_cap[..., 50]) + torch.mean(y_cap[..., 49])
                jrightankle_mean = torch.mean(
                    y_cap[..., 52]) + torch.mean(y_cap[..., 53]) + torch.mean(y_cap[..., 54])
                jrmrot_mean = torch.mean(
                    y_cap[..., 57]) + torch.mean(y_cap[..., 56]) + torch.mean(y_cap[..., 55])
                jrightfoot_mean = torch.mean(
                    y_cap[..., 58]) + torch.mean(y_cap[..., 59]) + torch.mean(y_cap[..., 60])
                jroot_mean = torch.mean(
                    y_cap[..., 63]) + torch.mean(y_cap[..., 62]) + torch.mean(y_cap[..., 61])

                jtrajectory_var = torch.sum(
                    (y_cap[..., 0] - jtrajectory_mean)**2) / (y_cap.shape[1]-1)
                jrightfoot_var = torch.sum(
                    (y_cap[..., 58] + y_cap[..., 59] + y_cap[..., 60] - jrightfoot_mean)**2) / (y_cap.shape[1]-1)
                jleftfoot_var = torch.sum(
                    (y_cap[..., 43] + y_cap[..., 44] + y_cap[..., 45] - jleftfoot_mean)**2) / (y_cap.shape[1]-1)
                jrightankle_var = torch.sum(
                    (y_cap[..., 52] + y_cap[..., 53] + y_cap[..., 54] - jrightankle_mean)**2) / (y_cap.shape[1]-1)
                jleftankle_var = torch.sum(
                    (y_cap[..., 37] + y_cap[..., 38] + y_cap[..., 39] - jleftankle_mean)**2) / (y_cap.shape[1]-1)
                jrightknee_var = torch.sum(
                    (y_cap[..., 49] + y_cap[..., 50] + y_cap[..., 51] - jrightknee_mean)**2) / (y_cap.shape[1]-1)
                jleftknee_var = torch.sum(
                    (y_cap[..., 34] + y_cap[..., 35] + y_cap[..., 36] - jleftknee_mean)**2) / (y_cap.shape[1]-1)
                jrighthip_var = torch.sum(
                    (y_cap[..., 46] + y_cap[..., 47] + y_cap[..., 48] - jrighthip_mean)**2) / (y_cap.shape[1]-1)
                jlefthip_var = torch.sum(
                    (y_cap[..., 31] + y_cap[..., 32] + y_cap[..., 33] - jlefthip_mean)**2) / (y_cap.shape[1]-1)
                jrightwrist_var = torch.sum(
                    (y_cap[..., 28] + y_cap[..., 29] + y_cap[..., 30] - jrightwrist_mean)**2) / (y_cap.shape[1]-1)
                jleftwrist_var = torch.sum(
                    (y_cap[..., 19] + y_cap[..., 20] + y_cap[..., 21] - jleftwrist_mean)**2) / (y_cap.shape[1]-1)
                jrightelbow_var = torch.sum(
                    (y_cap[..., 25] + y_cap[..., 26] + y_cap[..., 27] - jrightelbow_mean)**2) / (y_cap.shape[1]-1)
                jleftelbow_var = torch.sum(
                    (y_cap[..., 16] + y_cap[..., 17] + y_cap[..., 18] - jleftelbow_mean)**2) / (y_cap.shape[1]-1)
                jrightshoulder_var = torch.sum(
                    (y_cap[..., 22] + y_cap[..., 23] + y_cap[..., 24] - jrightshoulder_mean)**2) / (y_cap.shape[1]-1)
                jleftshoulder_var = torch.sum(
                    (y_cap[..., 13] + y_cap[..., 14] + y_cap[..., 15] - jleftshoulder_mean)**2) / (y_cap.shape[1]-1)
                jlmrot_var = torch.sum(
                    (y_cap[..., 40] + y_cap[..., 41] + y_cap[..., 42] - jlmrot_mean)**2) / (y_cap.shape[1]-1)
                jrmrot_var = torch.sum(
                    (y_cap[..., 55] + y_cap[..., 56] + y_cap[..., 57] - jrmrot_mean)**2) / (y_cap.shape[1]-1)

                jroot_var = torch.sum(
                    (y_cap[..., 61] + y_cap[..., 62] + y_cap[..., 63] - jroot_mean)**2) / (y_cap.shape[1]-1)
                jupperneck_var = torch.sum(
                    (y_cap[..., 12] + y_cap[..., 11] + y_cap[..., 10] - jupperneck_mean)**2) / (y_cap.shape[1]-1)
                jlowerneck_var = torch.sum(
                    (y_cap[..., 7] + y_cap[..., 8] + y_cap[..., 9] - jlowerneck_mean)**2) / (y_cap.shape[1]-1)
                jpelvis_var = torch.sum(
                    (y_cap[..., 1] + y_cap[..., 2] + y_cap[..., 3] - jpelvis_mean)**2) / (y_cap.shape[1]-1)
                jtorso_var = torch.sum(
                    (y_cap[..., 4] + y_cap[..., 5] + y_cap[..., 6] - jtorso_mean)**2) / (y_cap.shape[1]-1)

                trajectory_loss_ += math.sqrt(
                    mse_loss(jtrajectory_var, trajectory_var))
                pelvis_loss_ += math.sqrt(mse_loss(pelvis_var, jpelvis_var))
                torso_loss_ += math.sqrt(mse_loss(jtorso_var, torso_var))
                lowerneck_loss_ += math.sqrt(
                    mse_loss(jlowerneck_var, lowerneck_var))
                upperneck_loss_ += math.sqrt(
                    mse_loss(jupperneck_var, upperneck_var))
                leftshoulder_loss_ += math.sqrt(
                    mse_loss(leftshoulder_var, jleftshoulder_var))
                leftelbow_loss_ += math.sqrt(
                    mse_loss(jleftelbow_var, leftelbow_var))
                leftwrist_loss_ += math.sqrt(
                    mse_loss(leftwrist_var, jleftwrist_var))
                rightshoulder_loss_ += math.sqrt(
                    mse_loss(jrightshoulder_var, rightshoulder_var))
                rightelbow_loss_ += math.sqrt(
                    mse_loss(jrightelbow_var, rightelbow_var))
                rightwrist_loss_ += math.sqrt(
                    mse_loss(jrightwrist_var, rightwrist_var))
                lefthip_loss_ += math.sqrt(mse_loss(jlefthip_var, lefthip_var))
                leftknee_loss_ += math.sqrt(mse_loss(jleftknee_var,
                                                     leftknee_var))
                leftankle_loss_ += math.sqrt(
                    mse_loss(jleftankle_var, leftankle_var))
                lmrot_loss_ += math.sqrt(mse_loss(lmrot_var, jlmrot_var))
                leftfoot_loss_ += math.sqrt(mse_loss(jleftfoot_var,
                                                     leftfoot_var))

                righthip_loss_ += math.sqrt(mse_loss(jrighthip_var,
                                                     righthip_var))
                rightknee_loss_ += math.sqrt(
                    mse_loss(jrightknee_var, rightknee_var))
                rightankle_loss_ += math.sqrt(
                    mse_loss(jrightankle_var, rightankle_var))
                rmrot_loss_ += math.sqrt(mse_loss(rmrot_var, jrmrot_var))
                rightfoot_loss_ += math.sqrt(
                    mse_loss(jrightfoot_var, rightfoot_var))

                root_loss_ += math.sqrt(mse_loss(jroot_var, root_var))

                Tqdm.refresh()

                x = x.detach()
                y = y.detach()
                # loss = loss.detach()
                y_cap = y_cap.detach()

            if count >= 0 and args.debug:  # debugging by overfitting
                break

        total_loss = [trajectory_loss_, pelvis_loss_, torso_loss_, lowerneck_loss_,  upperneck_loss_,
                      leftshoulder_loss_, leftelbow_loss_, leftwrist_loss_, rightshoulder_loss_, rightelbow_loss_,
                      rightwrist_loss_, lefthip_loss_, leftknee_loss_, leftankle_loss_, leftfoot_loss_, righthip_loss_,
                      rightknee_loss_, rightankle_loss_, rightfoot_loss_, root_loss_]

        total_var = [jtrajectory_var, jpelvis_var, jtorso_var, jlowerneck_var,  jupperneck_var,
                     jleftshoulder_var, jleftelbow_var, jleftwrist_var, jrightshoulder_var, jrightelbow_var,
                     jrightwrist_var, jlefthip_var, jleftknee_var, jleftankle_var, jleftfoot_var, jrighthip_var,
                     jrightknee_var, jrightankle_var, jrightfoot_var, jroot_var]

        return total_loss, total_var

    # Sample
    ndev = len(dev)
    ntrain = len(train)
    ntest = len(test)

    test_loss_list, var_list = loop(model, data, test, pre, batch_size, 'test')
    test_loss_ = [x / ntest for x in test_loss_list]
    with open(dir_name+"/test_variance_error.bin", "wb") as fp:
        pickle.dump(test_loss_, fp)
    var_ = [x / ntest for x in var_list]
    with open(dir_name+"/test_variance.bin", "wb") as fp:
        pickle.dump(var_, fp)
    dev_loss_list = loop(model, data, dev, pre, batch_size, 'dev')
    dev_loss_ = [x / ndev for x in dev_loss_list]
    with open(dir_name+"/dev_variance.bin", "wb") as fp:
        pickle.dump(dev_loss_, fp)
    train_loss_list = loop(model, data, train, pre, batch_size, 'train')
    train_loss_ = [x / ntrain for x in train_loss_list]
    with open(dir_name+"/train_variance.bin", "wb") as fp:
        pickle.dump(train_loss_, fp)


if __name__ == '__main__':
    argparseNloop(sample)
