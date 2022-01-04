# DQBS: Deep Q-Learning with Backward SARSA

[2021 Deep Learning Course](http://da.inf.ethz.ch/teaching/2021/DeepLearning/) Project at ETH Zurich, [paper link](https://github.com/Jingyu6/dl_2021/blob/main/dqbs_paper.pdf)

### Authors:
By alphabetical order of last name \
[@Jingyu Liu](https://github.com/Jingyu6) \
[@Yunshu Ouyang](https://github.com/yooyoo9) \
[@Yilei Tu](https://github.com/yileitu) \
[@Yuyan Zhao](https://github.com/piew2)

### Abstract:
Deep Q-Network (DQN) with Experience Replay (ER) mechanism is the first algorithm to achieve super-human performance in Atari games and numerous works seek to improve upon them. Most previous works focused on designing architectures or updating rules for the Bellman updates to stabilize training. However, the bias of sampling transitions either randomly or weighted to some priority was often ignored and its validity taken for granted, as they usually perform well in practice. In this work, we designed a simple algorithm called Deep Q-Learning with Backward SARSA (DQBS) which splits a single standard update step of DQN into multiple steps with transitions following chronological backward order. DQBS takes advantage of the Markovian properties of transitions, which assume that the estimated Q-values of state-action pairs can gain more information from those of state-action pairs that follow immediately within trajectories. Each iteration now consists of a normal step which computes the Bellman targets with transitions sampled as usual and several backward steps which calculate the SARSA targets with transitions preceding the previous batch within trajectories. We justified the intuitions behind DQBS with illustrations, conducted ablation studies to prove that each design choice leads to a performance increase, and showed that DQBS outperforms DQN with ER in several Gym environments.

### Project Sturcture:
```
.
├── dqbs_paper.pdf
├── run.py <main script for experiment>
├── scripts <run experiments with the best parameters from ablation studies>
│   ├── run_cartpole.sh
│   └── run_acrobot.sh
│   └── run_mountaincar.sh
├── plots <plots used in paper>
│   ├── comparison
│       ├── ...
│   └── ablation
│       ├── ...
└── core <main implementation>
    ├── algorithms.py
    └── models.py
    └── replay_buffer.py
```

### How to Run Experiments:
To run experiments, go to the root directory and type (the default parameter can be used for result reproduction):
```
python run.py
```
The environment can be specified with **--env={cartpole,acrobot,mountaincar}** and the produced plots are stored in **./plots**.

To test the best parameters for DQBS chosen by ablation studies:
```
bash scripts/run_{cartpole, acrobot, mountaincar}.sh
```

### Contact information
We are looking forward to your feedback and advice: \
{liujin, ouyangy, yileitu, yuyzhao}@student.ethz.ch
