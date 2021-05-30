import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from nlp.lstm import BERTSentenceEncoder
from lossUtils import Loss
import pdb

import pickle as pkl
import numpy as np


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class Integrator(nn.Module):
    '''
    A velocity integrator.
    If we have displacement values for translation and such, and we know the exact timesteps of the signal, 
    we can calculate the global values efficiently using a convolutional layer with weights set to 1 and kernel_size=timesteps

    Note: this method will not work for realtime scenarios. Although, it is efficient enough to keep adding displacements over time
    '''

    def __init__(self, channels, time_steps):
        super(Integrator, self).__init__()
        self.conv = CausalConv1d(in_channels=channels,
                                 out_channels=channels,
                                 kernel_size=time_steps,
                                 stride=1,
                                 dilation=1,
                                 groups=channels,
                                 bias=False)
        self.conv.weight = nn.Parameter(torch.ones_like(
            self.conv.weight), requires_grad=False)

    def forward(self, xs):
        return self.conv(xs)


class TeacherForcing():
    '''
    Sends True at the start of training, i.e. Use teacher forcing maybe.
    Progressively becomes False by the end of training, start using gt to train
    '''

    def __init__(self, max_epoch):
        self.max_epoch = max_epoch

    def __call__(self, epoch, batch_size=1):
        p = epoch*1./self.max_epoch
        random = torch.rand(batch_size)
        return (p < random).double()

# Sequence to Sequence AutoEncoder


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size,
                          num_layers=num_layers, batch_first=True, dropout=0.2)

    def forward(self, x):
        outputs, h_n = self.rnn(x)
        # TODO include attention
        return outputs


class TrajectoryPredictor(nn.Module):
    def __init__(self, pose_size, trajectory_size, hidden_size):
        super(TrajectoryPredictor, self).__init__()
        self.lp = nn.Linear(hidden_size, pose_size)
        self.fc = nn.Linear(pose_size+hidden_size, trajectory_size)

    def forward(self, x):
        pose_vector = self.lp(x)
        trajectory_vector = self.fc(torch.cat((pose_vector, x), dim=-1))
        mixed_vector = torch.cat((trajectory_vector, pose_vector), dim=-1)
        return mixed_vector


class RotDecoderCell(nn.Module):
    def __init__(self, hidden_size, pose_size, trajectory_size, use_h=False, use_tp=True, use_lang=False):
        super(RotDecoderCell, self).__init__()
        self.use_h = 1 if use_h else 0
        self.use_lang = 1 if use_lang else 0
        self.rnn = nn.GRUCell(input_size=pose_size+trajectory_size+hidden_size*(self.use_h+self.use_lang),
                              hidden_size=hidden_size)
        if use_tp:
            self.tp = TrajectoryPredictor(pose_size=pose_size,
                                          trajectory_size=trajectory_size,
                                          hidden_size=hidden_size)
        else:
            self.tp = nn.Linear(hidden_size, pose_size + trajectory_size)

        if self.use_lang:
            self.lin = nn.Linear(hidden_size+pose_size +
                                 trajectory_size, pose_size+trajectory_size)

    def forward(self, x, h):
        if self.use_h:
            x_ = torch.cat([x, h], dim=-1)
        else:
            x_ = x
        h_n = self.rnn(x_, h)
        # TODO add attention
        tp_n = self.tp(h_n)

        return tp_n, h_n


class VelDecoderCell(nn.Module):
    def __init__(self, hidden_size, pose_size, trajectory_size, use_h=False, use_tp=True, use_lang=False):
        super(VelDecoderCell, self).__init__()
        self.use_h = 1 if use_h else 0
        self.use_lang = 1 if use_lang else 0
        self.rnn = nn.GRUCell(input_size=pose_size+trajectory_size+hidden_size*(self.use_h+self.use_lang),
                              hidden_size=hidden_size)
        if use_tp:
            self.tp = TrajectoryPredictor(pose_size=pose_size,
                                          trajectory_size=trajectory_size,
                                          hidden_size=hidden_size)
        else:
            self.tp = nn.Linear(hidden_size, pose_size + trajectory_size)

        if self.use_lang:
            self.lin = nn.Linear(hidden_size+pose_size +
                                 trajectory_size, pose_size+trajectory_size)

    def forward(self, x, h):
        if self.use_h:
            x_ = torch.cat([x, h], dim=-1)
        else:
            x_ = x
        h_n = self.rnn(x_, h)
        # TODO add attention
        tp_n = self.tp(h_n)
        if self.use_lang:
            y = self.lin(x) + tp_n
        else:
            y = x + tp_n
        return y, h_n


class VelDecoder(nn.Module):
    def __init__(self, hidden_size, pose_size, trajectory_size,
                 use_h=False, start_zero=False, use_tp=True,
                 use_lang=False, use_attn=False):
        super(VelDecoder, self).__init__()

        self.input_size = pose_size + trajectory_size
        self.cellvel = VelDecoderCell(hidden_size, pose_size, trajectory_size,
                                      use_h=use_h, use_tp=use_tp, use_lang=use_lang)
        # Hardcoded to reach 0% Teacher forcing in 10 epochs
        self.tf = TeacherForcing(0.1)
        self.start_zero = start_zero
        self.use_lang = use_lang
        self.use_attn = use_attn

    def forward(self, h, time_steps, gt, epoch=np.inf, attn=None):
        if self.use_lang:
            lang_z = h
        if self.start_zero:
            print('start_zero')
            x = h.new_zeros(h.shape[0], self.input_size)
            x = h.new_tensor(torch.rand(h.shape[0], self.input_size))
        else:
            x = gt[:, 0, :]  # starting point for the decoding
        xrot = x
        xvel = x
        h1 = h
        Y = []
        for t in range(time_steps):
            if self.use_lang:
                if self.use_attn:  # calculate attention at each time-step
                    lang_z = attn(h)
                    x, h = self.cellvel(torch.cat([x, lang_z], dim=-1), h)

            else:
                x, h = self.cellvel(x, h)

            Y.append(x.unsqueeze(1))
            if t > 0:
                mask = self.tf(epoch, h.shape[0]).double(
                ).view(-1, 1).to(x.device)
                x = mask * gt[:, t-1, :] + (1-mask) * x
        return torch.cat(Y, dim=1)

    def sample(self, h, time_steps, start, attn=None):
        if self.use_lang:
            lang_z = h

        #x = torch.rand(h.shape[0], self.input_size).to(h.device).to(h.dtype)
        x = start  # starting point for the decoding
        Y = []
        for t in range(time_steps):
            if self.use_lang:
                if self.use_attn:
                    lang_z = attn(h)
                x, h = self.cellvel(torch.cat([x, lang_z], dim=-1), h)
            else:
                x, h = self.cellvel(x, h)
            Y.append(x.unsqueeze(1))
        return torch.cat(Y, dim=1)


class RotDecoder(nn.Module):
    def __init__(self, hidden_size, pose_size, trajectory_size,
                 use_h=False, start_zero=False, use_tp=True,
                 use_lang=False, use_attn=False):
        super(RotDecoder, self).__init__()

        self.input_size = pose_size + trajectory_size
        self.cellrot = RotDecoderCell(hidden_size, pose_size, trajectory_size,
                                      use_h=use_h, use_tp=use_tp, use_lang=use_lang)

        # Hardcoded to reach 0% Teacher forcing in 10 epochs
        self.tf = TeacherForcing(0.1)
        self.start_zero = start_zero
        self.use_lang = use_lang
        self.use_attn = use_attn

    def forward(self, h, time_steps, gt, epoch=np.inf, attn=None):
        if self.use_lang:
            lang_z = h
        if self.start_zero:
            print('start_zero')
            x = h.new_zeros(h.shape[0], self.input_size)
            x = h.new_tensor(torch.rand(h.shape[0], self.input_size))
        else:
            x = gt[:, 0, :]  # starting point for the decoding
        xrot = x
        xvel = x
        h1 = h
        Y = []
        for t in range(time_steps):
            if self.use_lang:
                if self.use_attn:  # calculate attention at each time-step
                    lang_z = attn(h)
                    x, h = self.cellrot(torch.cat([x, lang_z], dim=-1), h)

            else:
                x, h = self.cellrot(x, h)

            Y.append(x.unsqueeze(1))
            if t > 0:
                mask = self.tf(epoch, h.shape[0]).double(
                ).view(-1, 1).to(x.device)
                x = mask * gt[:, t-1, :] + (1-mask) * x
        return torch.cat(Y, dim=1)

    def sample(self, h, time_steps, start, attn=None):
        if self.use_lang:
            lang_z = h

        #x = torch.rand(h.shape[0], self.input_size).to(h.device).to(h.dtype)
        x = start  # starting point for the decoding
        Y = []
        for t in range(time_steps):
            if self.use_lang:
                if self.use_attn:
                    lang_z = attn(h)
                x, h = self.cellrot(torch.cat([x, lang_z], dim=-1), h)
            else:
                x, h = self.cellrot(x, h)
            Y.append(x.unsqueeze(1))
        return torch.cat(Y, dim=1)


class Seq2Seq(nn.Module):
    # TODO add an integrator and differentiator to handle delta values end2end
    def __init__(self, hidden_size, pose_size, trajectory_size,
                 use_h=False, start_zero=False, use_tp=True,
                 use_lang=False, use_attn=False, **kwargs):
        super(Seq2Seq, self).__init__()
        if use_attn:  # use_lang must be true if use_attn is true
            use_lang = True
        # TODO take root rotation out of Trajectory Predictor
        #pose_size += 4
        #trajectory_size -= 4
        input_size = pose_size + trajectory_size
        self.enc = Encoder(input_size, hidden_size)
        # self.rotdec = RotDecoder(hidden_size, pose_size, trajectory_size,
        #                    use_h=use_h, start_zero=start_zero,
        #                    use_tp=use_tp, use_lang=use_lang,
        #                    use_attn=use_attn)
        self.veldec = VelDecoder(hidden_size, pose_size, trajectory_size,
                                 use_h=use_h, start_zero=start_zero,
                                 use_tp=use_tp, use_lang=use_lang,
                                 use_attn=use_attn)

    def forward(self, x, train=True, epoch=np.inf, attn=None):
        time_steps = x.shape[1]
        enc_vector = self.enc(x)[:, -1, :]
        dec_rot = self.rotdec(enc_vector, time_steps,
                              gt=x, epoch=epoch, attn=attn)
        dec_vel = self.veldec(enc_vector, time_steps,
                              gt=x, epoch=epoch, attn=attn)
        return dec_rot, dec_vel, []


class PoseGenerator(nn.Module):

    def __init__(self, chunks, input_size=300, Seq2SeqKwargs={}, load=None):
        super(PoseGenerator, self).__init__()
        self.chunks = chunks

        Seq2SeqKwargs['pose_size'] = 63
        self.hidden_size = Seq2SeqKwargs['hidden_size']
        self.trajectory_size = Seq2SeqKwargs['trajectory_size']
        self.pose_size = Seq2SeqKwargs['pose_size']
        self.seq2seq = Seq2Seq(**Seq2SeqKwargs)
        # self.sentence_enc = LSTMEncoder(self.hidden_size)
        self.sentence_enc = BERTSentenceEncoder(self.hidden_size)
        if load:
            self.load_state_dict(torch.load(open(load, 'rb')))
            print('PoseGenerator Model Loaded')
        else:
            print('PoseGenerator Model initialising randomly')

    def forward(self, P_in, gt, s2v, skel=None, train=False, epoch=np.inf):
        z_in_pose = self.seq2seq.enc(P_in)
        language_z, _ = self.sentence_enc(s2v)
        # z_in = torch.cat((z_in_pose[:, -1, :], language_z), -1)
        time_steps = P_in.shape[-2]
        # Q_r_lang = self.seq2seq.rotdec(language_z, time_steps, gt=P_in)
        Q_v_lang = self.seq2seq.veldec(language_z, time_steps, gt=P_in)
        # Q_r = self.seq2seq.rotdec(z_in_pose[:,-1,:], time_steps, gt=P_in)
        Q_v = self.seq2seq.veldec(z_in_pose[:, -1, :], time_steps, gt=P_in)
        # P_gen_lang = 0.5 *(Q_r_lang + Q_v_lang)
        # P_gen = 0.5 *(Q_r + Q_v)

        # z_gen_r = self.seq2seq.enc(Q_r)
        z_gen_v = self.seq2seq.enc(Q_v)

        manifold_loss = F.smooth_l1_loss(z_gen_v, z_in_pose, reduction='mean')
        encoder_loss = F.smooth_l1_loss(
            z_in_pose[:, -1, :], language_z, reduction='mean')
        reconstruction_loss = F.smooth_l1_loss(
            Q_v, gt, reduction='mean') + F.smooth_l1_loss(Q_v_lang, gt, reduction='mean')
        velocity_orig = P_in[:, 1:, :] - P_in[:, :-1, :]
        velocity_Q_v = Q_v[:, 1:, :] - Q_v[:, :-1, :]
        velocity_Q_v_lang = Q_v_lang[:, 1:, :] - Q_v_lang[:, :-1, :]

        velocity_loss = F.smooth_l1_loss(velocity_orig, velocity_Q_v) + \
            F.smooth_l1_loss(velocity_orig, velocity_Q_v_lang)
        # + F.smooth_l1_loss(velocity_orig, velocity_Q_r) + \
        # F.smooth_l1_loss(velocity_orig, velocity_Q_r_lang)
        internal_losses = [0.001*manifold_loss, 0.1 *
                           encoder_loss, reconstruction_loss, 0.1*velocity_loss]

        return Q_v, internal_losses

    def sample(self, s2v, time_steps, start):
        start = start[..., :-4]
        language_z, _ = self.sentence_enc(s2v)
        Q_v_lang = self.vel_dec.sample(language_z, time_steps, start)
        tz = torch.zeros((Q_v_lang.shape[0], Q_v_lang.shape[1], 4)).to(
            Q_v_lang.device).double()

        predicted_pose = torch.cat((Q_v_lang, tz), dim=-1)
        print(predicted_pose.shape)
        return predicted_pose, []

    def sample_encoder(self, s2v, x):
        ''' last four columns of P_in is always 0 in input so no need to train it. Just ouput zero for columns 63,64,65, 66'''
        P_in = x[..., :-4]
        z_in_pose = self.seq2seq.enc(P_in)
        language_z, _ = self.sentence_enc(s2v)
        z = z_in_pose[:, -1, :]
        gs = language_z * torch.transpose(language_z, 0, 1)
        gp = z * torch.transpose(z, 0, 1)
        return z, language_z, gp, gs


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(64, 64, num_layers=1, batch_first=True)
        self.net = nn.Sequential(nn.Linear(64, 256),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Linear(256, 128),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Linear(128, 64),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Linear(64, 1),
                                 nn.Sigmoid()
                                 )

    def forward(self, x):
        x, h = self.rnn(x)
        out = self.net(h)
        return out.squeeze(0)
