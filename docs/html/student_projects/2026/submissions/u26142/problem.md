# Proximal Policy Optimization (PPO) from scratch

## Problem

In reinforcement learning, a *policy* chooses actions from observations to
earn rewards. PPO trains a policy from batches of experience while limiting
how much its action probabilities change per update. While PPO can be applied
to robot control, games, language-model post-training, as well as various other
problems, this program uses navigation as an example for demonstrating PPO.

One robot navigates a randomly generated square grid. It observes its position,
the goal's absolute position, and the full obstacle map. It moves in four
directions. Reaching the goal gives +1, each step costs 0.01, and hitting a
wall costs another 0.02. There
is no reward for getting closer. `--size` sets the grid width
(default 5), `--obstacles` defaults to 30% of its area, and `--horizon`
defaults to `size*size` steps. Each start and goal are distinct and connected
by a route within that limit. A shortest-path search checks reachability when
generating maps but does not guide the robot.

## How the serial baseline code works

`ppo.cc` uses only the C++17 standard library. It runs `--envs` independent
environments **sequentially**, collecting `--rollout` steps per environment
with a fixed policy. Each step records the observation, action, reward, and
the policy's action probability and value prediction (expected future reward).
One rollout may cross an episode boundary.

The program uses generalized advantage estimation (GAE) to estimate whether
actions did better than predicted; it stops the calculation at episode ends
and uses a value prediction if a rollout ends mid-episode. It then shuffles
the batch and trains for `--epochs` passes in groups of `--minibatch` steps.
PPO clips its training objective when an action's probability ratio moves
too far from the probability recorded during collection.
The shared model has ReLU hidden layers (`--num-layers` layers of `--hidden`
units) and separate action-probability and value outputs. Gradients and Adam
updates are implemented directly, without a machine-learning library.

Every `--log-every` updates, the program prints and flushes `training_s`
(cumulative rollout and learning time, excluding evaluation), `train_return`
(mean reward of recently finished training episodes), `eval_return` (mean
reward on separate fixed-seed maps), and `eval_success` (fraction reaching the
goal). Evaluation uses the most probable action and does not train the model.

In the notebook's compilation cell, after `cd code` and module setup:

```sh
clang++ -O3 -Wall -std=c++17 ppo.cc -o ppo
```

In its `%%sub` run cell, after `cd code`:

```sh
./ppo --size 5 --updates 20000 --envs 16 --rollout 16 --epochs 3 --minibatch 128 --hidden 64 --num-layers 2 --log-every 1000
```

An additional `--seed` parameter sets the random seed for reproducibility.

## Where is parallelism?

In addition to the usual GPU acceleration of neural-network matrix operations,

- Independent environments can generate experience on different CPU cores.
- Some environments could be implemented entirely as vectorized operations, allowing
  many instances to run in parallel on a GPU.
- GAE can also run independently across rollouts, though steps within each rollout
  have a backward-in-time dependency.
