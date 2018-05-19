import sys
sys.path.append('/home/zhangjunhao/options')
from base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.is_Train = True
        self.parser.add_argument('--batchsize', type=int, default=3, help='input batch size')
        self.parser.add_argument('--lr_G', type=float, default=0.0002, help='initial learning rate of Generator')
        self.parser.add_argument('--lr_D', type=float, default=0.0002, help='initial learning rate of Discriminator')
        self.parser.add_argument('--count_epoch', type=int, default=0, help='the starting count epoch count')
        self.parser.add_argument('--epochs', type=int, default=10000, help='number of epochs for train')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='adam optimizer parameter')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='adam optimizer parameter')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving the latest results')
        self.parser.add_argument('--D_interval', type=int, default=20, help='the interval of each optimization of D')
        self.parser.add_argument('--w_L1', type=int, default=1, help='the weight of the L1 loss')
