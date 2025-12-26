export PYTHONPATH=.
python tools/GaPaTTA.py --ema_rate=0.999 --model_lr=0.0001 --prompt_lr=0.0001 --prompt_sparse_rate=0.1 --scale=0.1

wait
