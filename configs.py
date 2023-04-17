in_batch_size: 16
validation_batch_size: 32
max_epochs: 500
num_tta_iterations: 20
in_channels: 12
num_layers_freeze: -1

early_stop:
patience: 25

optimizer:
    name: 'Adam'
    lr: 0.0001
    momentum: 0.9
    weight_decay: 0.0001

lr_scheduler:
    name: 'CustomScheduler'
    factor: 0.9
    gamma: 0.1
    min_lr: 0.0000001
    patience: 20
    verbose: True
    change_epoch: 25
