import torch
import pandas as pd
from utils.kit_visualization import render, render4
from common.parallel import parallel
from argsUtils import argparseNloop
from utils.quaternion import *
import numpy as np
import pdb
from pathlib import Path


def fill_missing_joints(df, joints):
    joints_in_df = set([col[:-3] for col in df.columns])
    missing_joints = set(joints) - joints_in_df
    missing_xyz = ['{}_{}'.format(jnt, xyz) for jnt in missing_joints for xyz in [
        'rx', 'ry', 'rz']]
    missing_w = ['{}_rw'.format(jnt) for jnt in missing_joints]

    df_missing_xyz = pd.DataFrame(data=np.zeros(
        (df.shape[0], len(missing_xyz))), columns=missing_xyz)
    df_missing_w = pd.DataFrame(data=np.ones(
        (df.shape[0], len(missing_w))), columns=missing_w)

    return pd.concat([df, df_missing_w, df_missing_xyz], axis=1)


def quat2xyz(df, skel):
    # df = fill_missing_joints(df, skel.joints)
    root_pos = torch.from_numpy(
        df[['root_tx', 'root_ty', 'root_tz']].values).unsqueeze(0)
    columns = [str_format.format(joint) for joint in skel.joints for str_format in [
        '{}_rw', '{}_rx', '{}_ry', '{}_rz']]
    root_orientation = torch.from_numpy(df[columns].values)
    root_orientation = root_orientation.view(
        1, root_orientation.shape[0], -1, 4)
    xyz_data = skel.forward_kinematics(root_orientation, root_pos)[0].numpy()
    return xyz_data


def rifke2xyz(M, data):
    # df = fill_missing_joints(df, skel.joints)
    output_columns = data.raw_data.output_columns('rifke')
    # M[output_columns].to_csv(filename.with_suffix('.rifke').as_posix()) ## save rifke as well
    xyz_data = data.raw_data.rifke2fke(M[output_columns].values)

    return xyz_data


def axis2xyz(df, skel, filename=None, dataset=None, subset=None):
    # df = fill_missing_joints(df, skel.joints)
    root_pos = torch.from_numpy(
        df[['root_tx', 'root_ty', 'root_tz']].values).unsqueeze(0).to(device='cuda')
    columns = [str_format.format(joint) for joint in skel.joints for str_format in [
        '{}_rx', '{}_ry', '{}_rz']]
    rotations = torch.from_numpy(df[columns].values).to(device='cuda')
    rotations = rotations.view(rotations.shape[0], -1, 3)
    # Axis angle to quaternion
    root_orientation = expmap_to_quaternion(rotations.cpu().detach().numpy())
    root_orientation = torch.from_numpy(
        root_orientation).to(device='cuda').unsqueeze(0)
    # skel.save_as_bvh(root_orientation[0], root_pos[0], str(filename).zfill(6), dataset_name=dataset, subset_name=subset)
    xyz_data = skel.forward_kinematics(root_orientation, root_pos)[
        0].cpu().numpy()
    return xyz_data


def readNrender(filenum, filename, description, skel, time, output, figsize, feats_kind, data):
    # filenum, filename, description, skel, time, output, figsize, feats_kind = params
    df = pd.read_csv(filename, index_col=0)
    print("inside readNRender")
    if feats_kind == 'quaternion':
        xyz_data = quat2xyz(df, skel)
    elif feats_kind == 'rifke':
        xyz_data = rifke2xyz(df, data)
    elif feats_kind == 'fke':
        xyz_data = df.values.reshape(df.shape[0], -1, 3)

    render(xyz_data, skel, time, output, figsize, description)
    print(filenum, filename)


def chunk_description(description, max_len=40):
    description = description.split(' ')
    chunks = []
    chunk = ''
    length = 0
    for desc in description:
        length += len(desc) + 1
        if length > max_len:
            chunks.append(chunk)
            length = len(desc) + 1
            chunk = desc + ' '
        else:
            chunk += desc + ' '
    if chunk:
        chunks.append(chunk)
    description = '\n'.join(chunks)
    return description


def get_description(df, filename, path2data, feats_kind):
    filename = (Path(path2data)/filename).as_posix()
    try:
        # if df[df[feats_kind].any() == filename] is not None:
        description = df[df[feats_kind] == filename].iloc[0]['descriptions']
    except:
        description = []
    else:
        description = chunk_description(description)

    return description


def parallelRender(filenames, descriptions, outputs, skel, feats_kind, data=None):
    filenums = [i for i in range(len(filenames))]
    skels = [skel for _ in range(len(filenames))]

    times = [np.inf for _ in range(len(filenames))]
    figsizes = [(4, 4) for _ in range(len(filenames))]
    feats_kind = [feats_kind] * len(filenames)

    print(filenums)
    params = zip(filenums, filenames, descriptions, skels,
                 times, outputs, figsizes, feats_kind)
    if not descriptions:
        print('description is empty.')
        descriptions = filenums
    for i in range(len(filenames)):
        print("The number of file is  ", i)
        readNrender(filenums[i], filenames[i], descriptions[i],
                    skels[i], times[i], outputs[i], figsizes[i], feats_kind[i], data)


def renderOnSlurm(dir_name, dataset, feats_kind):
    pass
