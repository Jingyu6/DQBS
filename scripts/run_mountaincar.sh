# !/bin/bash

echo 'Start running experiments for mountaincar'
python run.py --env=mountaincar --lr=5e-4 --epochs=400 --buffer_size=1e4 --lr=5e-4 --buffer_size=1e4
python run.py --env=mountaincar --lr=1e-4 --epochs=400 --buffer_size=1e4 --use_prioritized_buffer --lr=1e-4 --buffer_size=1e4 --beta=1e-2 --alpha=0.2
