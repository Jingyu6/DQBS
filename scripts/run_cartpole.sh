# !/bin/bash

echo 'Start running experiments for cartpole'
python run.py --lr=5e-3 --buffer_size=1e5
python run.py --lr=5e-3 --buffer_size=1e5 --use_prioritized_buffer --backtrack_steps=5 --beta=1e-4 --alpha=0.4
