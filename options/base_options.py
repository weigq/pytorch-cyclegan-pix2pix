"""
running basic options
"""

import argparse
import os
from util import util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.parser.add_argument('--dataroot',         required=True, help='path to images')
        self.parser.add_argument('--batchSize',        type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize',         type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize',         type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc',         type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc',        type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--ngf',              type=int, default=64, help='# of G filters in 1st conv layer')
        self.parser.add_argument('--ndf',              type=int, default=64, help='# of D filters in 1st conv layer')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks', help='selects model for netG')
        self.parser.add_argument('--n_layers_D',       type=int, default=3, help='used if which_model_netD==n_layers')
        self.parser.add_argument('--gpu_ids',          type=str, default='0', help='gpu ids ')
        self.parser.add_argument('--name',             type=str, default='experiment_name', help='name of experiment')
        self.parser.add_argument('--dataset_mode',     type=str, default='unaligned', help='datasets loaded type [unaligned | aligned | single]')
        self.parser.add_argument('--model',            type=str, default='cycle_gan', help='chooses model. cycle_gan, pix2pix, test')
        self.parser.add_argument('--which_direction',  type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads',         default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir',  type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm',             type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches',   action='store_true', help='whether data shuffle')
        self.parser.add_argument('--display_winsize',  type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id',       type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--display_port',     type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')

        # self.parser.add_argument('--identity',         type=float, default=0.0, help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
        self.parser.add_argument('--no_dropout',       action='store_true', help='no dropout for the generator')

        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Max number of samples allowed per dataset')
        self.parser.add_argument('--resize_or_crop',   type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip',          action='store_true', help='if specified, do not flip the images for data augmentation')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        gpu_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for gpu in gpu_ids:
            gpu = int(gpu)
            if gpu >= 0:
                self.opt.gpu_ids.append(gpu)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
