"""
training options
"""

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        BaseOptions.__init__(self)
        self.isTrain = None

    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq',     type=int, default=100, help='freq of showing results on screen')
        self.parser.add_argument('--print_freq',       type=int, default=100, help='freq of showing results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='freq of saving latest results')
        self.parser.add_argument('--save_epoch_freq',  type=int, default=5, help='freq of saving ckpt at end of epochs')
        self.parser.add_argument('--continue_train',   action='store_true', help='continue training')
        self.parser.add_argument('--epoch_count',      type=int, default=1, help='the starting epoch count')
        self.parser.add_argument('--phase',            type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch',      type=str, default='latest', help='which epoch to load')
        self.parser.add_argument('--niter',            type=int, default=100, help='iter at starting lr')
        self.parser.add_argument('--niter_decay',      type=int, default=100, help='iter to linearly decay lr to zero')
        self.parser.add_argument('--beta1',            type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr',               type=float, default=0.0002, help='initial lr for adam')
        self.parser.add_argument('--no_lsgan',         action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--lambda_A',         type=float, default=10.0, help='weight of cycle loss(A->B->A)')
        self.parser.add_argument('--lambda_B',         type=float, default=10.0, help='weight of cycle loss(B->A->B)')
        self.parser.add_argument('--pool_size',        type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html',          action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.isTrain = True
