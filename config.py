import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'lr': (0.1, 'learning rate'),
        'cuda': (0, 'which cuda device to use (-1 for cpu training)'),
        'dropout': (0.0, 'dropout probability'),
        'epochs': (100, 'maximum number of epochs to train for'),
        'weight-decay': (0., 'l2 regularization strength'),
        'optimizer': ('Adam', 'which optimizer to use'),
        'momentum': (0.999, 'momentum in optimizer'),
        'patience': (500, 'patience for early stopping'),
        'seed': (1234, 'seed for training'),
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'test-freq': (1, 'how often to compute test metrics (in epochs)'),
        'save': (0, '1 to save model and logs and 0 otherwise'),
        'save-dir': (None, 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
        'gamma': (0.5, 'gamma for lr scheduler'),
        'print-epoch': (True, ''),
        'grad-clip': (None, 'max norm for gradient clipping, or None for no gradient clipping'),
        'min-epochs': (100, 'do not early stop before min-epochs'),
        'batch_size': (32, 'the batch size'),
        'num_of_weeks': (0, 'number of weeks'),
        'num_of_days': (0, 'number of days'),
        'num_of_hours': (1, 'number of hours'),
        'start_epoch': (0, 'start epoch, if not 0, load the parameters of the specified epoch'),
        'in_channels': (1, 'the dimension of the input feature'),
        't_kernel': (3, 'size of time kernel'),
        'use_att': (True, 'whether to use attention'),
    },
    'model_config': {
        'task': ('nr', 'which tasks to train on, can be any of [nr, lp, nc]'),
        'model': ('LSTGCN', 'our model'),
        'dim': ('[4]', 'list for embedding dimension for each layer. Example: [16,16,32,32]'),
        'models_dim': ('[4]', 'list for embedding dimension for each model'),
        'c': (1.0, 'hyperbolic radius, set to None for trainable curvature'),
        'r': (2., 'fermi-dirac decoder parameter for lp'),
        't': (1., 'fermi-dirac decoder parameter for lp'),
        'pretrained-embeddings': (None, 'path to pretrained embeddings'),
        'pos-weight': (0, 'whether to upweight positive class in node classification tasks'),
        'num-layers': (4, 'number of hidden layers in encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'double-precision': ('0', 'whether to use double precision')
    },
    'data_config': {
        'dataset_name': ('PEMS03', 'which dataset to use'),
        'num_of_vertices': (358, 'number of nodes in the dataset'),
    },
    'my_config': {
        'variables': ([], 'parameters'),
        'lr_scheduler': ('step', 'which scheduler to use'),
        'margin': (2., 'margin of MarginLoss'),
        'lr_gamma': (0.98, 'gamma for scheduler'),
        'step_lr_gamma': (0.4, 'gamma for StepLR scheduler'),
        'step_lr_reduce_freq': (30, 'step size for StepLR scheduler'),
        'weight_decay': (0.0, 'weight decay'),
        'proj_init': ('xavier', 'the way to initialize parameters'),
        'embed_manifold': ('euclidean', ''),
        "select_manifold": ('lorentz', 'selected manifold'),
        "num_centroid": (4, 'number of centroids'),
        "feature_dim": (1, "input feature dimensionality",),
        "pre_trained":(False, "whether use pre-train model"),
        "tie_weight": (0, "whether to tie transformation matrices"),
        "hyp_rotation": (1,"whether to include hyperbolic rotation to spatial feature transformation"),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
