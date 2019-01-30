import time
import sys
from tensorboardX import SummaryWriter
sys.path.append('options')
from train_options import TrainOptions
sys.path.append('data')
from data_loader import CreateDataLoader
sys.path.append('model')
from model_Loader import CreateModel
sys.path.append('util')
from utils import error as err

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
model = CreateModel(opt)

writer = SummaryWriter('logs')

err = err(model.save_dir)
for epoch in range(opt.count_epoch + 1,  opt.epochs + 1):
    epoch_start_time = time.time()
    err.initialize()

    for i, data in enumerate(data_loader):
        model.forward(data)

        model.optimize_G_parameters()
        if(i % opt.D_interval == 0):
            model.optimize_D_parameters()

        err.add(model.Loss_G.data.item(), model.Loss_D.data.item())

    LOSSG, LOSSD = err.print_errors(epoch)
    writer.add_scalar('loss_g', LOSSG, epoch)
    writer.add_scalar('loss_d', LOSSD, epoch)
    print('End of epoch {0} \t Time Taken: {1} sec\n'.format(epoch, time.time()-epoch_start_time))
    model.save_result(epoch)
    if epoch % opt.save_epoch_freq == 0:
        print('Saving the model at the end of epoch {}\n'.format(epoch))
        model.save(epoch)
