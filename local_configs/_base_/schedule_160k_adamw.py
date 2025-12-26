# # optimizer
# optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.0001)
# optimizer_config = dict()
# # learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
# # runtime settings
# runner = dict(type='IterBasedRunner', max_iters=160000)
# checkpoint_config = dict(by_epoch=False, interval=4000)
# evaluation = dict(interval=4000, metric='mIoU')

# optimizer
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.0001)
optimizer_config = dict()

# learning policy
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=0.0,
    by_epoch=True  # ✅ 开启以 epoch 为单位的调度
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)  # ✅ 修改为 epoch 模式

checkpoint_config = dict(by_epoch=True, interval=5)  # ✅ 每5个 epoch 存一次
evaluation = dict(interval=1, metric='mIoU')  # ✅ 每个 epoch 评估一次


