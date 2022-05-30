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
                    split=split, batch_size=batch_size,
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
        # checkpoint = torch.load(args.load)
        # model.load_state_dict(checkpoint['model'])

        # model_dict = checkpoint['model']
        # for param_tensor in model_dict:
        # 	print(param_tensor, "\t")
        # 
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

        # root_rot_loss_ = 0
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
                trajectory_loss_ += math.sqrt(mse_loss(y_cap[:,:, 0], y[:,:, 0]))
                pelvis_loss_ += math.sqrt(mse_loss(y_cap[:, :, 1], y[:, :, 1])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 2], y[:, :, 2])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 3], y[:, :, 3]))
                torso_loss_ += math.sqrt(mse_loss(y_cap[:, :, 4], y[:, :, 4])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 5], y[:, :, 5])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 6], y[:, :, 6]))
                lowerneck_loss_ += math.sqrt(mse_loss(y_cap[:, :, 7], y[:, :, 8])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 8], y[:, :, 8])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 9], y[:, :, 9]))
                upperneck_loss_ += math.sqrt(mse_loss(y_cap[:, :, 10], y[:, :, 10])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 11], y[:, :, 11])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 12], y[:, :, 12]))
                leftshoulder_loss_ += math.sqrt(mse_loss(y_cap[:, :, 13], y[:, :, 13])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 14], y[:, :, 14])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 15], y[:, :, 15]))
                leftelbow_loss_ += math.sqrt(mse_loss(y_cap[:, :, 16], y[:, :, 16])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 17], y[:, :, 17])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 18], y[:, :, 18]))
                leftwrist_loss_ += math.sqrt(mse_loss(y_cap[:, :, 19], y[:, :, 19])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 20], y[:, :, 20])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 21], y[:, :, 21]))
                rightshoulder_loss_ += math.sqrt(mse_loss(y_cap[:, :, 22], y[:, :, 22])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 23], y[:, :, 23])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 24], y[:, :, 24]))
                rightelbow_loss_ += math.sqrt(mse_loss(y_cap[:, :, 25], y[:, :, 25])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 26], y[:, :, 26])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 27], y[:, :, 27]))
                rightwrist_loss_ += math.sqrt(mse_loss(y_cap[:, :, 28], y[:, :, 28])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 29], y[:, :, 29])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 30], y[:, :, 30]))
                lefthip_loss_ += math.sqrt(mse_loss(y_cap[:, :, 31], y[:, :, 31])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 32], y[:, :, 32])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 33], y[:, :, 33]))

                leftknee_loss_ += math.sqrt(mse_loss(y_cap[:, :, 34], y[:, :, 34])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 35], y[:, :, 35])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 36], y[:, :, 36]))
                leftankle_loss_ += math.sqrt(mse_loss(y_cap[:, :, 37], y[:, :, 37])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 38], y[:, :, 38])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 39], y[:, :, 39]))
                leftfoot_loss_ += math.sqrt(mse_loss(y_cap[:, :, 43], y[:, :, 43])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 44], y[:, :, 44])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 45], y[:, :, 45]))

                righthip_loss_ += math.sqrt(mse_loss(y_cap[:, :, 46], y[:, :, 46])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 47], y[:, :, 47])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 48], y[:, :, 48]))

                rightknee_loss_ += math.sqrt(mse_loss(y_cap[:, :, 49], y[:, :, 49])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 50], y[:, :, 50])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 51], y[:, :, 51]))
                rightankle_loss_ += math.sqrt(mse_loss(y_cap[:, :, 52], y[:, :, 52])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 53], y[:, :, 53])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 54], y[:, :, 54]))

                rightfoot_loss_ += math.sqrt(mse_loss(y_cap[:, :, 58], y[:, :, 58])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 59], y[:, :, 59])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 60], y[:, :, 60]))
                root_loss_ += math.sqrt(mse_loss(y_cap[:, :, 61], y[:, :, 61])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 62], y[:, :, 62])) + \
                    math.sqrt(mse_loss(y_cap[:, :, 63], y[:, :, 63]))
                # mean_pose_loss_ += math.sqrt(mse_loss(y_cap[:,:,1:], y[:,:,1:]))

                # running_mean_pose_count +=  (kount+1) * 67
                running_joint_count += (kount+1) * 3

                # if criterion.loss_list:
                #   loss = criterion(y_cap, y)
                #   loss_ = loss.item()
                # else:
                #   loss = 0
                #   loss_ = 0
                # for i_loss in internal_losses:
                #   loss += i_loss
                #   loss_ += i_loss.item()
                #   running_internal_loss += i_loss.item()

                # running_loss += loss_
                # running_count +=  np.prod(y.shape)
                # update tqdm
                # Tqdm.set_description(desc+' {:.4f} {:.4f}'.format(running_loss/running_count, running_internal_loss/running_count))
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
         rightknee_loss_, rightankle_loss_, rightfoot_loss_, root_loss_ ]

        return total_loss

    # Sample
    ndev = len(dev)
    ntrain = len(train)
    ntest = len(test)
    # dev_loss_list = loop(model, data, dev, pre, batch_size, 'dev')
    # dev_loss_ = [x / ndev for x in dev_loss_list]
    # with open(dir_name+"/deverror_global.bin", "wb") as fp:
    #     pickle.dump(dev_loss_, fp)

    test_loss_list = loop(model, data, test, pre, batch_size, 'test')
    test_loss_ = [x / ntest for x in test_loss_list]
    with open(dir_name+"/testerror_global.bin", "wb") as fp:
        pickle.dump(test_loss_, fp)

    # train_loss_list = loop(model, data, train, pre, batch_size, 'train')
    # train_loss_ = [x / ntrain for x in train_loss_list]
    # with open(dir_name+"/trainerror_global.bin", "wb") as fp:
    #     pickle.dump(train_loss_, fp)
    # baseline_loss_sum = list(
    #     map(add, list(map(add, dev_loss_list, test_loss_list)), train_loss_list))
    # total_datas = len(dev) + len(test) + len(train)
    # baseline_loss = [x / total_datas for x in baseline_loss_sum]
    # with open(dir_name+"/APE_total.bin", "wb") as fp:
    #     pickle.dump(baseline_loss, fp)


if __name__ == '__main__':
    argparseNloop(sample)
