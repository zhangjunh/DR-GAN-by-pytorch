import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import numpy as np
import gc
import torch.cuda

from Component import Generator
from Component import Discriminator
from Component import one_hot
from Component import weights_init_normal
from Component import Tensor2Image
from base_model import BaseModel


class Single_DRGAN(BaseModel):
    """
    The model of Single_DRGAN according to the options.
    """

    def name(self):
        return 'Single_DRGAN'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.is_Train = opt.is_Train

        self.G = Generator(N_p=opt.N_p, N_z=opt.N_z, single=True)
        self.D = Discriminator(N_p=opt.N_p, N_d=opt.N_d)
        if self.is_Train:
            self.optimizer_G = optim.Adam(self.G.parameters(), lr=opt.lr_G, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = optim.Adam(self.D.parameters(), lr=opt.lr_D, betas=(opt.beta1, opt.beta2))
            self.criterion = nn.CrossEntropyLoss()
            self.L1_criterion = nn.L1Loss()
            self.w_L1 = opt.w_L1

        self.N_z = opt.N_z
        self.N_p = opt.N_p
        self.N_d = opt.N_d

    def init_weights(self):
        self.G.apply(weights_init_normal)
        self.D.apply(weights_init_normal)

    def load_input(self, input):
        self.image = []
        self.pose = []
        self.identity = []
        self.name = []
        for i in range(len(input['pose'])):
            self.image.append(input['image'][i])
            self.pose.append(input['pose'][i])
            self.identity.append(input['identity'][i])
            self.name.append(input['name'][i])

    def set_input(self, input):
        """
        the structure of the input.
        {
        'image':Bx3x96x96 FloatTensor
        'pose':Bx1 FloatTensor
        'identity':Bx1 FloatTensor
        }

        test_pose (B): used for the test='initial learning rate
        """
        self.load_input(input)
        self.image = torch.squeeze(torch.stack(self.image, dim = 0))
        self.batchsize = len(self.pose)
        self.pose = torch.LongTensor(self.pose)
        self.frontal_pose = torch.LongTensor(np.random.randint(self.N_p, size = self.batchsize))

        # if self.is_Train:
        self.input_pose = one_hot(self.frontal_pose, self.N_p)
        # else:
        #     self.input_pose = one_hot(test_pose.long(), self.N_p)

        self.identity = torch.LongTensor(self.identity)
        self.fake_identity = torch.zeros(self.batchsize).long() # 0 indicates fake
        self.noise = torch.FloatTensor(np.random.normal(loc=0.0, scale=0.3, size=(self.batchsize, self.N_z)))

        #cuda
        if self.opt.gpu_ids:
            self.image = self.image.cuda()
            self.pose = self.pose.cuda()
            self.frontal_pose = self.frontal_pose.cuda()
            self.input_pose = self.input_pose.cuda()
            self.identity = self.identity.cuda()
            self.fake_identity = self.fake_identity.cuda()
            self.noise = self.noise.cuda()

        self.image = Variable(self.image)
        self.pose = Variable(self.pose)
        self.frontal_pose = Variable(self.frontal_pose)
        self.input_pose = Variable(self.input_pose)
        self.identity = Variable(self.identity)
        self.fake_identity = Variable(self.fake_identity)
        self.noise = Variable(self.noise)

    def forward(self, input):
        self.set_input(input)

        self.syn_image = self.G(self.image, self.input_pose, self.noise)
        self.syn = self.D(self.syn_image)
        self.syn_identity = self.syn[:, :self.N_d+1]
        self.syn_pose = self.syn[:, self.N_d+1:]

        self.real = self.D(self.image)
        self.real_identity = self.real[:, :self.N_d+1]
        self.real_pose = self.real[:, self.N_d+1:]

    def backward_G(self):
        self.Loss_G_syn_identity = self.criterion(self.syn_identity, self.identity)
        self.Loss_G_syn_pose = self.criterion(self.syn_pose, self.frontal_pose)
        self.L1_Loss = self.L1_criterion(self.syn_image, self.image)

        self.Loss_G = self.Loss_G_syn_identity + self.Loss_G_syn_pose + self.w_L1 * self.L1_Loss
        self.Loss_G.backward(retain_graph=True)

    def backward_D(self):
        self.Loss_D_real_identity = self.criterion(self.real_identity, self.identity)
        self.Loss_D_real_pose = self.criterion(self.real_pose, self.pose)

        self.Loss_D_syn = self.criterion(self.syn_identity, self.fake_identity)

        self.Loss_D = self.Loss_D_real_identity + self.Loss_D_real_pose + self.Loss_D_syn
        self.Loss_D.backward()

    def optimize_G_parameters(self):
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def optimize_D_parameters(self):
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def print_current_errors(self):
        print('Loss_G: {0} \t Loss_D: {1}'.format(self.Loss_G.data[0], self.Loss_D.data[0]))

    def save(self, epoch):
        self.save_network(self.G, 'G', epoch, self.gpu_ids)
        self.save_network(self.D, 'D', epoch, self.gpu_ids)

    def save_result(self, epoch=None):
        for i, syn_img in enumerate(self.syn_image.data):
            img = self.image.data[i]
            filename = self.name[i]

            if epoch:
                filename = 'epoch{0}_{1}'.format(epoch, filename)

            path = os.path.join(self.result_dir, filename)
            img = Tensor2Image(img)
            syn_img = Tensor2Image(syn_img)

            width, height = img.size
            result_img = Image.new(img.mode, (width*2, height))
            result_img.paste(img, (0, 0, width, height))
            result_img.paste(syn_img, box=(width, 0))
            result_img.save(path)

class Multi_DRGAN(BaseModel):
    """
    The model of Multi_DRGAN according to the options.
    """
    def name(self):
        return 'Multi_DRGAN'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.is_Train = opt.is_Train

        self.G = Generator(N_p=opt.N_p, N_z=opt.N_z)
        self.D = Discriminator(N_p=opt.N_p, N_d=opt.N_d)
        if self.is_Train:
            self.optimizer_G = optim.Adam(self.G.parameters(), lr=opt.lr_G, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = optim.Adam(self.D.parameters(), lr=opt.lr_D, betas=(opt.beta1, opt.beta2))
            self.criterion = nn.CrossEntropyLoss()
            self.L1_criterion = nn.L1Loss()
            self.w_L1 = opt.w_L1

        self.N_z = opt.N_z
        self.N_p = opt.N_p
        self.N_d = opt.N_d

        import torchvision.models as models
        self.resnet18 = models.resnet18(pretrained=True)
        if self.gpu_ids:
            self.resnet18 = self.resnet18.cuda()
        self.resnet18.fc = torch.nn.LeakyReLU(0.1)
        for param in self.resnet18.parameters():
            param.requires_grad = False
        self.resnet18.eval()

        torch.set_num_threads(1)

    def init_weights(self):
        self.G.apply(weights_init_normal)
        self.D.apply(weights_init_normal)

    def load_input(self, input):
        self.image = []
        self.pose = []
        self.identity = []
        self.name = []
        for i in range(len(input)):
            self.image.append(input[i]['image'])
            self.pose.append(input[i]['pose'])
            self.identity.append(input[i]['identity'])
            self.name.append(input[i]['name'])
        # self.image = [item for sublist in self.image for item in sublist]
        # self.pose = [item for sublist in self.pose for item in sublist]
        # self.identity = [item for sublist in self.identity for item in sublist]
        # self.name = [item for sublist in self.name for item in sublist]
        del input
        gc.collect()

    def set_input(self, input):
        """
        the structure of the input.
        {
        'image':Bx3x96x96 FloatTensor
        'pose':Bx1 FloatTensor
        'identity':Bx1 FloatTensor
        }

        test_pose (B): used for the test='initial learning rate
        """
        self.load_input(input)
        self.batchsize = []
        self.rand_pose = []
        self.frontal_pose = []
        self.input_pose = []
        self.fake_identity = []
        self.noise = []
        self.frontal_image = [[] for r in range(len(self.pose))]
        self.profile_image = [[] for r in range(len(self.pose))]
        # self.real_frontal_pose = [[] for r in range(len(self.pose))]
        # self.real_profile_pose = [[] for r in range(len(self.pose))]
        # self.frontal_id = [[] for r in range(len(self.pose))]
        self.profile_id = [[] for r in range(len(self.pose))]
        self.frontal_name = [[] for r in range(len(self.pose))]
        self.profile_name = [[] for r in range(len(self.pose))]
        for j in range(len(self.pose)):
            for e in range(len(self.pose[j])):
                if self.pose[j][e]:
                    self.frontal_image[j].append(self.image[j][e])
                    # self.real_frontal_pose[j].append(self.pose[j][e])
                    # self.frontal_id[j].append(self.identity[j][e])
                    self.frontal_name[j].append(self.name[j][e])
                else:
                    self.profile_image[j].append(self.image[j][e])
                    # self.real_profile_pose[j].append(self.pose[j][e])
                    self.profile_id[j].append(self.identity[j][e])
                    self.profile_name[j].append(self.name[j][e])
            self.frontal_image[j] = torch.squeeze(torch.stack(self.frontal_image[j], dim=0))
            self.profile_image[j] = torch.squeeze(torch.stack(self.profile_image[j], dim=0))
            # self.real_frontal_pose[j] = torch.LongTensor(self.real_frontal_pose[j])
            # self.real_profile_pose[j] = torch.LongTensor(self.real_profile_pose[j])
            # self.frontal_id[j] = torch.LongTensor(self.frontal_id[j])
            self.profile_id[j] = torch.LongTensor(self.profile_id[j])
            x = torch.split(self.identity[j], 1)
            self.identity[j] = torch.cat((self.identity[j], x[0]), 0)
            if self.is_Train:
                self.batchsize.append((self.pose[j]).size(0))
            else:
                self.batchsize.append((self.profile_id[j]).size(0))
            self.rand_pose.append(torch.LongTensor(np.random.randint(self.N_p, size=self.batchsize[j] + 1)))
            self.frontal_pose.append(torch.ones(self.batchsize[j] + 1).long())
            x = torch.split(self.profile_id[j], 1)
            self.profile_id[j] = torch.cat((self.profile_id[j], x[0]), 0)
            if self.is_Train:
                self.input_pose.append(one_hot(self.rand_pose[j], self.N_p))
            else:
                self.input_pose.append(one_hot(self.frontal_pose[j], self.N_p))

            self.fake_identity.append(torch.zeros(self.batchsize[j] + 1).long()) # 0 indicates fake
            self.noise.append(torch.FloatTensor(np.random.normal(loc=0.0, scale=0.3, size=(self.batchsize[j] + 1, self.N_z))))

            #cuda
            if self.opt.gpu_ids:
                self.image[j] = self.image[j].cuda()
                self.pose[j] = self.pose[j].cuda()
                self.rand_pose[j] = self.rand_pose[j].cuda()
                self.frontal_pose[j] = self.frontal_pose[j].cuda()
                self.input_pose[j] = self.input_pose[j].cuda()
                self.identity[j] = self.identity[j].cuda()
                self.fake_identity[j] = self.fake_identity[j].cuda()
                self.noise[j] = self.noise[j].cuda()
                self.frontal_image[j] = self.frontal_image[j].cuda()
                self.profile_image[j] = self.profile_image[j].cuda()
                # self.real_frontal_pose[j] = self.real_frontal_pose[j].cuda()
                # self.real_profile_pose[j] = self.real_profile_pose[j].cuda()
                # self.frontal_id[j] = self.frontal_id[j].cuda()
                self.profile_id[j] = self.profile_id[j].cuda()

            self.image[j] = Variable(self.image[j])
            self.pose[j] = Variable(self.pose[j])
            self.rand_pose[j] = Variable(self.rand_pose[j])
            self.frontal_pose[j] = Variable(self.frontal_pose[j])
            self.input_pose[j] = Variable(self.input_pose[j])
            self.identity[j] = Variable(self.identity[j])
            self.fake_identity[j] = Variable(self.fake_identity[j])
            self.noise[j] = Variable(self.noise[j])
            self.frontal_image[j] = Variable(self.frontal_image[j])
            self.profile_image[j] = Variable(self.profile_image[j])
            # self.real_frontal_pose[j] = Variable(self.real_frontal_pose[j])
            # self.real_profile_pose[j] = Variable(self.real_profile_pose[j])
            # self.frontal_id[j] = Variable(self.frontal_id[j])
            self.profile_id[j] = Variable(self.profile_id[j])

    def forward(self, input):
        self.set_input(input)
        self.syn_image = []
        self.syn = []
        self.syn_identity = []
        self.syn_pose = []
        self.real = []
        self.real_identity = []
        self.real_pose = []

        for k in range(len(self.pose)):
            if self.is_Train:
                self.syn_image.append(self.G(self.image[k], self.input_pose[k], self.noise[k]))
            else:
                self.syn_image.append(self.G(self.profile_image[k], self.input_pose[k], self.noise[k]))
            self.syn.append(self.D(self.syn_image[k]))
            self.syn_identity.append(self.syn[k][:, :self.N_d+1])
            self.syn_pose.append(self.syn[k][:, self.N_d+1:])

            self.real.append(self.D(self.image[k]))
            self.real_identity.append(self.real[k][:, :self.N_d+1])
            self.real_pose.append(self.real[k][:, self.N_d+1:])

    def feature_loss(self, syn_image, real_image):
        import torchvision.transforms as transforms
        real_image = real_image.expand_as(syn_image)
        self.syn_img = []
        self.real_img = []
        # for s in range(len(syn_image)):
        transform_loss = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Scale((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.syn_img.append(transform_loss(syn_image.cpu().data))
        self.real_img.append(transform_loss(real_image.cpu().data))
        self.syn_img = torch.stack(self.syn_img, 0)
        self.real_img = torch.stack(self.real_img, 0)
        if self.gpu_ids:
            self.syn_img = self.syn_img.cuda()
            self.real_img = self.real_img.cuda()
        self.syn_img = Variable(self.syn_img, requires_grad=True)
        self.real_img = Variable(self.real_img, requires_grad=False)
        self.result_syn_img = self.resnet18(self.syn_img)
        self.result_real_img = self.resnet18(self.real_img)
        # syn_img = syn_img.data.cpu().numpy()
        # real_img = real_img.data.cpu().numpy()
        # result = np.mean(np.absolute(np.subtract(syn_img, real_img)))
        # result = torch.mean(torch.abs(syn_img.sub(real_img)))
        return self.result_syn_img, self.result_real_img

    def backward_G(self):
        self.Loss_G_syn_identity = []
        self.Loss_G_syn_pose = []
        self.Loss_feature = []
        self.Loss_G = []
        for l in range(len(self.pose)):

            # self.Loss_G_syn_identity.append(self.criterion(self.syn_identity[l], self.profile_id[l]))
            # self.Loss_G_syn_pose.append(self.criterion(self.syn_pose[l], self.frontal_pose[l]))
            self.Loss_G_syn_identity.append(self.criterion(self.syn_identity[l], self.identity[l]))
            self.Loss_G_syn_pose.append(self.criterion(self.syn_pose[l], self.rand_pose[l]))
            if not self.frontal_image[l].size() == torch.Size([3,96,96]):
                self.sum_feature_loss = []
                for k in range(len(self.image[l]) + 1):
                    if self.rand_pose[l][k].data.cpu().numpy():
                        self.result_syn_img, self.result_real_img = self.feature_loss(self.syn_image[l][k], self.frontal_image[l][np.random.randint(len(self.frontal_name[l]))])
                    # self.result_real_img = self.result_real_img.detach()
                        self.sum_feature_loss.append(self.L1_criterion(self.result_syn_img, self.result_real_img))
                if len(self.sum_feature_loss):
                    self.Loss_feature.append(sum(self.sum_feature_loss) / len(self.sum_feature_loss))
                else:
                    self.Loss_feature.append(self.Loss_G_syn_pose[l])
            else:
                self.sum_feature_loss = []
                for k in range(len(self.image[l]) + 1):
                    if self.rand_pose[l][k].data.cpu().numpy():
                        self.result_syn_img, self.result_real_img = self.feature_loss(self.syn_image[l][k], self.frontal_image[l])
                    # self.result_real_img = self.result_real_img.detach()
                        self.sum_feature_loss.append(self.L1_criterion(self.result_syn_img, self.result_real_img))
                if len(self.sum_feature_loss):
                    self.Loss_feature.append(sum(self.sum_feature_loss) / len(self.sum_feature_loss))
                else:
                    self.Loss_feature.append(self.Loss_G_syn_pose[l])
            # else:
            #     self.result_syn_img, self.result_real_img = self.feature_loss(self.syn_image[l], self.frontal_image[l])
            #     # self.result_real_img = self.result_real_img.detach()
            #     self.Loss_feature.append(self.L1_criterion(self.result_syn_img, self.result_real_img))
            self.Loss_G.append(self.Loss_G_syn_identity[l] + self.Loss_G_syn_pose[l] + self.w_L1 * self.Loss_feature[l])
        self.Loss_G = sum(self.Loss_G)
        self.Loss_G.backward(retain_graph=True)

    def backward_D(self):
        self.Loss_D_real_identity = []
        self.Loss_D_real_pose = []
        self.Loss_D_syn = []
        self.Loss_D = []
        for m in range(len(self.pose)):
            y = torch.split(self.identity[m], self.batchsize[m])
            self.Loss_D_real_identity.append(self.criterion(self.real_identity[m], y[0]))
            self.Loss_D_real_pose.append(self.criterion(self.real_pose[m], self.pose[m]))
            self.Loss_D_syn.append(self.criterion(self.syn_identity[m], self.fake_identity[m]))

            self.Loss_D.append(self.Loss_D_real_identity[m] + self.Loss_D_real_pose[m] + self.Loss_D_syn[m])
        self.Loss_D = sum(self.Loss_D)
        self.Loss_D.backward()

    def optimize_G_parameters(self):
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def optimize_D_parameters(self):
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def print_current_errors(self):
        print('Loss_G: {0} \t Loss_D: {1}'.format(self.Loss_G.data[0], self.Loss_D.data[0]))

    def save(self, epoch):
        self.save_network(self.G, 'G', epoch, self.gpu_ids)
        self.save_network(self.D, 'D', epoch, self.gpu_ids)

    def save_result(self, epoch=None):
        for w in range(len(self.pose)):
            for i, syn_img in enumerate(self.syn_image[w].data):
                if self.is_Train:
                    if i == self.batchsize[w]:
                        img = self.image[w].data[-1]
                        filename = 'mixed_' + self.name[w][-1]
                    else:
                        img = self.image[w].data[i]
                        filename = self.name[w][i]
                else:
                    if i == self.batchsize[w]:
                        if not self.frontal_image[w].size() == torch.Size([3, 96, 96]):
                            img = self.frontal_image[w].data[0]
                        else:
                            img = self.frontal_image[w].data
                        filename = 'mixed_' + self.frontal_name[w][0]
                    else:
                        img = self.profile_image[w].data[i]
                        filename = self.profile_name[w][i]

                if epoch:
                    filename = 'epoch{0}_{1}'.format(epoch, filename)

                path = os.path.join(self.result_dir, filename)
                img = Tensor2Image(img)
                syn_img = Tensor2Image(syn_img)

                width, height = img.size
                result_img = Image.new(img.mode, (width*2, height))
                result_img.paste(img, (0, 0, width, height))
                result_img.paste(syn_img, box=(width, 0))
                result_img.save(path)
