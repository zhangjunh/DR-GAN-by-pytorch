import sys
sys.path.append('/home/zhangjunhao/options')
from base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.is_Train = False
        self.parser.add_argument('--batchsize', type=int, default=3, help='input batch size')
