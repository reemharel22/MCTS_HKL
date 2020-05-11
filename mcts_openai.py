#!/usr/bin/env python2
import os
import gym
import sys
import random
import itertools
import copy
from math import sqrt, log
import HklEnv
import baselines
import numpy as np


def moving_average(v, n):
    n = min(len(v), n)
    ret = [.0]*(len(v)-n+1)
    ret[0] = float(sum(v[:n]))/n
    for i in range(len(v)-n):
        ret[i+1] = ret[i] + float(v[n+i] - v[i])/n
    return ret


def softmax(x):
    probs = np.exp(x - np.max(x))
    # https://mp.weixin.qq.com/s/2xYgaeLlmmUfxiHCbCa8dQ
    # avoid float overflow and underflow
    probs /= np.sum(probs)
    return probs


def ucb(node):
    return node.value / node.visits + sqrt(log(node.parent.visits)/node.visits)


def combinations(space):
    if isinstance(space, baselines.spaces.Bin_Discrete):
        return np.where(space.mask == 0)[0].tolist()
    elif isinstance(space, gym.spaces.Discrete):
        return range(space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return itertools.product(*[combinations(s) for s in space.spaces])
    else:
        raise NotImplementedError


class Node:
    counter = 0

    def __init__(self, parent, action, directory):
        self.id = Node.counter = Node.counter+1
        self.parent = parent
        self.action = action
        self.children = []
        self.explored_children = 0
        self.visits = 0
        self.reward = 0
        self.value = 0
        self.state = ""

    def to_json(self):
        ret_dict = {}
        ret_dict['id'] = self.id
        ret_dict['action'] = self.action
        ret_dict['visits'] = self.visits
        ret_dict['value'] = self.value
        ret_dict['reward'] = self.reward
        ret_dict['state'] = self.state
        ret_dict['children'] = [None] * 4
        for child in self.children:
            ret_dict['children'][child.action] = child.to_json()
        return ret_dict


class Runner:
    def __init__(self, rec_dir, env_name, max_depth=1000, playouts=10000):
        self.env_name = env_name
        self.max_depth = max_depth
        self.playouts = playouts
        self.dir = rec_dir+'/'+env_name
        os.makedirs(self.dir)
        self.root = Node(None, None, self.dir)
        self.env = gym.make(self.env_name)

    def print_stats(self, loop, score, avg_time):
        sys.stdout.write('\r%3d   score:%10.3f   avg_time:%4.1f s' % (loop, score, avg_time))
        sys.stdout.flush()

    def reset_player(self):
        self.root = Node(None, None, self.dir)

    # TODO: start the playout just from the given env state (originally, started from root).
    def get_action(self,env):
        for _ in range(self.playouts):
            env_state = copy.copy(env)

            sum_reward = 0
            node = self.root
            terminal = False
            actions = []

            # selection
            while node.children:
                if node.explored_children < len(node.children):
                    child = node.children[node.explored_children]
                    node.explored_children += 1
                    node = child
                else:
                    node = max(node.children, key=ucb)
                _, reward, terminal, info = env_state.step(node.action)
                node.state = str(info['hkl'][0]) + " " + str(info['hkl'][1]) + " " + str(info['hkl'][2])
                node.reward = reward
                sum_reward += reward
                actions.append(node.action)

            # expansion
            if not terminal:
                node.children = [Node(node, a, self.dir) for a in combinations(env_state.action_space)]
                random.shuffle(node.children)

            # playout
            while not terminal:
                action = env_state.action_space.sample()
                _, reward, terminal, _ = env_state.step(action)
                sum_reward += reward
                actions.append(action)

                if len(actions) > self.max_depth:
                    sum_reward -= 100
                    break

            # backpropagate
            while node:
                node.visits += 1
                node.value += sum_reward
                node = node.parent

        # The following lines are adopted form AlphaGo to calculate the probabilities of each step.
        move_probs = np.zeros(len(self.root.children))
        act_visits = [(node.action, node.visits)
                      for node in self.root.children]
        acts, visits = zip(*act_visits)
        # TODO: We may consider replace softmax with UCB.
        p = softmax(1.0 / 1.0 * np.log(np.array(visits) + 1e-10))
        move_probs[list(acts)] = p
        move = np.random.choice(acts, p=move_probs)

        return move, move_probs


def main():
    # get rec_dir
    if not os.path.exists('rec'):
        os.makedirs('rec')
        next_dir = 0
    else:
        next_dir = max([int(f) for f in os.listdir('rec')+["0"] if f.isdigit()])+1
    rec_dir = 'rec/'+str(next_dir)
    os.makedirs(rec_dir)
    print ("rec_dir:", rec_dir)

    runner = Runner(rec_dir, 'hkl-v1', max_depth=10000, playouts=22*22*22)
    move, probs = runner.get_action(runner.env)
    print(move, probs)
    runner.env.step(move)

    # Toy text
    # Runner(rec_dir, 'NChain-v0', loops=100, playouts=3000, max_depth=50).run()
    #
    # # Algorithmic
    # Runner(rec_dir, 'Copy-v0').run()
    # Runner(rec_dir, 'RepeatCopy-v0').run()
    # Runner(rec_dir, 'DuplicatedInput-v0').run()
    # Runner(rec_dir, 'ReversedAddition-v0').run()
    # Runner(rec_dir, 'ReversedAddition3-v0').run()
    # Runner(rec_dir, 'Reverse-v0').run()


if __name__ == "__main__":
    main()
