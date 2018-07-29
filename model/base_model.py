import os
import torch


class BaseModel(object):

	def initialize(self, opt):
		self.opt = opt
		self.gpu_ids = opt.gpu_ids
		self.is_Train = opt.is_Train
		self.save_dir = os.path.join(opt.checkpoints_dir, opt.model)
		self.result_dir = os.path.join(opt.test_dir, opt.pretrained_G) if opt.pretrained_G else opt.test_dir

	# helper saving function that can be used by subclasses
	def save_network(self,network, network_label, epoch_label, gpu_ids):
		save_filename = '{0}_net_{1}.path'.format(epoch_label, network_label)
		save_path = os.path.join(self.save_dir, save_filename)
		torch.save(network.cpu().state_dict(), save_path)
		if len(gpu_ids) and torch.cuda.is_available():
			network.cuda(device=gpu_ids[0])

	def load(self, network, filename):
		# save_path = os.path.join(self.save_dir, filename)
		save_path = os.path.join('checkpoints', filename)
		network.load_state_dict(torch.load(save_path))


	# for the use of the breakpoint
	def reload(self, count_epoch):
		G_filename = '{}_net_G.path'.format(count_epoch)
		D_filename = '{}_net_D.path'.format(count_epoch)
		self.load(self.G, G_filename)
		self.load(self.D, D_filename)

	# update learning rate  (called once every epoch)
	def update_learning_rate(self):
		for scheduler in self.schedulers:
			scheduler.step()
		lr = self.optimizers[0].param_groups[0]['lr']
		print('learning rate = %.7f'%lr)
