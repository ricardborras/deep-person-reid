import torchreid

datamanager = torchreid.data.ImageDataManager(
    root='/home/counterest/datasets',
    sources='busreid',
    targets='busreid',
    height=128,
    width=128,
    batch_size_train=64,
    batch_size_test=300,
    transforms=['random_flip', 'random_erase']
)

model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=True
)

optimizer = torchreid.optim.build_optimizer(
    model,
    optim='amsgrad',
    lr=0.0015
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='cosine'
)

model = model.cuda()

torchreid.utils.load_pretrained_weights(model, 'log/osnet_x1_0_dukemtmcreid_softmax_cosinelr/model.pth.tar-250')

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)

engine.run(
    save_dir='log/bus_reid',
    test_only=True,
    visrank=True,
)

