import time
import sys
import tensorflow as tf
sys.path.append('/home/zhangjunhao/options')
from train_options import TrainOptions
sys.path.append('/home/zhangjunhao/data')
from data_loader import CreateDataLoader
sys.path.append('/home/zhangjunhao/model')
from model_Loader import CreateModel
sys.path.append('/home/zhangjunhao/util')
from utils import error as err

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
model = CreateModel(opt)

sess = tf.Session()
loss_g = tf.placeholder(tf.float32)
loss_d = tf.placeholder(tf.float32)
me1 = tf.summary.scalar('loss_g', loss_g)
me2 = tf.summary.scalar('loss_d', loss_d)
merged = tf.summary.merge([me1, me2])
writer = tf.summary.FileWriter("/home/zhangjunhao/logs", sess.graph)

err = err(model.save_dir)
for epoch in range(opt.count_epoch + 1,  opt.epochs + 1):
    epoch_start_time = time.time()
    err.initialize()

    for i, data in enumerate(data_loader):
        model.forward(data)

        model.optimize_G_parameters()
        if(i % opt.D_interval == 0):
            model.optimize_D_parameters()

        err.add(model.Loss_G.data[0], model.Loss_D.data[0])

    LOSSG, LOSSD = err.print_errors(epoch)
    summary = sess.run(merged, feed_dict={loss_g: LOSSG, loss_d: LOSSD})
    writer.add_summary(summary, epoch)
    print('End of epoch {0} \t Time Taken: {1} sec\n'.format(epoch, time.time()-epoch_start_time))
    model.save_result(epoch)
    if epoch % opt.save_epoch_freq == 0:
        print('Saving the model at the end of epoch {}\n'.format(epoch))
        model.save(epoch)
