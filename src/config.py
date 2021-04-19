DUMP_FOLDER = 'dumps/'
SAVE_WEIGHTS_PATH = DUMP_FOLDER + 'weights.pth'
SAVE_METRICS_PATH = DUMP_FOLDER + 'metrics.pkl'

DATASET_PATHS = {
    'main_path': 'imagewoof2/',
    'train': 'imagewoof2/train/',
    'val': 'imagewoof2/val/',
    'labels': 'imagewoof2/noisy_imagewoof.csv'
}

CNFG = {
    'seed': 42,
    'device': 'cuda:0',
    'img_size': 256,
    'batch_size': 40,
    'lr': 1e-3,

    'multi_gpu': False,
    'epochs': 100,
    'esr': 5,
    'num_workers': 0,
    'model_arch': 'mobilenetv2_100', #'tf_efficientnet_b4_ns',
    'grad_clip': 5
}
