# !/bin/bash

echo 'Start running experiments for acrobot'
python run.py --env=acrobot --lr=5e-4 --buffer_size=1e3
python run.py --env=acrobot --lr=1e-4 --buffer_size=1e3 --use_prioritized_buffer --beta=1e-4 --alpha=0.4
