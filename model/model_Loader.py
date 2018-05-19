import torch

def CreateModel(opt):
    model = None
    if opt.model == 'single':
        from DRGAN import Single_DRGAN
        model = Single_DRGAN()
    elif opt.model == 'multi':
        from DRGAN import Multi_DRGAN
        model = Multi_DRGAN()
    else:
        raise ValueError('Model {} not recongnized.'.format(opt.model))

    model.initialize(opt)
    model.init_weights()

    if opt.pretrained_D:
        model.load(model.D, opt.pretrained_D)
    if opt.pretrained_G:
        model.load(model.G, opt.pretrained_G)

    if opt.is_Train and opt.count_epoch:
        model.reload(opt.count_epoch)

    if len(opt.gpu_ids) and torch.cuda.is_available():
        model.G.cuda()
        model.D.cuda()
        if opt.is_Train:
            model.criterion.cuda()
            model.L1_criterion.cuda()

    print('model {} was created'.format(model.name()))
    return model
