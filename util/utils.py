from torchvision import transforms
from PIL import Image
import os

class error(object):
    """
    the class to calculate the error of every epoch.
    """
    def __init__(self, dir):
        filename = 'error_record.txt'
        self.path = os.path.join(dir, filename)
        if os.path.exists(self.path):
            open(self.path, 'w').close()    #clear the content of the txt file.
        else:
            os.mknod(self.path)


    def initialize(self):
        self.Loss_G = 0
        self.Loss_D = 0
        self.count = 0

    def add(self, Loss_G, Loss_D):
        self.Loss_G += Loss_G
        self.Loss_D += Loss_D
        self.count += 1

    def calculate(self):
        self.Loss_G /= self.count
        self.Loss_D /= self.count

    def print_errors(self, epoch):
        self.calculate()
        with open(self.path, 'a') as f:
            f.write('epoch{0}:\tLoss_G: {1}\tLoss_D: {2}\n'.format(epoch, self.Loss_G, self.Loss_D))
        print('Loss_G: {0}\tLoss_D: {1}'.format(self.Loss_G, self.Loss_D))
        return self.Loss_G, self.Loss_D
