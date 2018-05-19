from options.train_options import TrainOptions
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from data.data_processing import show_sample

train_opt = TrainOptions().parse()
train_dataloader = CreateDataLoader(train_opt)
test_opt = TestOptions().parse()
test_dataloader = CreateDataLoader(test_opt)
loader = iter(train_dataloader)
for i in range(100):
    batch = loader.next()
# img = batch['image']
# pose = batch['pose']
# identity = batch['identity']
# assert len(train_dataloader) == 6300
s = batch
show_sample(s)
# _use_shared_memory = False
# assert len(test_dataloader) == 700        imgs = make_dataset(root, class_to_idx)

# from model.model_Loader import CreateModel
# model = CreateModel(train_opt)
#
# model.forward(batch)
# model.backward_D()
# model.backward_G()

# model.Loss_D.backward()
# model.Loss_D_real_identity.backward()
# model.Loss_D_real_pose.backward()
# model.Loss_D_fake.backward()

# model.Loss_G.backward()
# model.Loss_G_fake_pose.backward()
# model.Loss_G_fake_identity.backward()

# model.optimize_G_parameters()
# model.optimize_D_parameters()
# model.print_current_errors()
# model.save(1)
# model.load(model.G, '1_net_G.path')
# model.load(model.D, '1_net_D.path')
###########################
import matplotlib.pyplot as plt
import numpy as np
