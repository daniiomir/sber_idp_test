DATASET_PATHS = {
    'main_path': 'imagewoof2/',
    'train': 'imagewoof2/train/',
    'test': 'imagewoof2/test/',
    'labels': 'imagewoof2/noisy_imagewoof.csv'
}

CNFG = {
    'seed': 42,
    'img_size': 400,
    'batch_size': 10,
    'lr': 1e-3,

    'multi_gpu': False,
    'epochs': 100,
    'esr': 5,
    'num_workers': 0,
    'model_arch': 'tf_efficientnet_b4_ns',
    'device': 'cpu',
    # 'tta': 5,
    'grad_clip': 5
}
