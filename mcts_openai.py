#!/usr/bin/env python2
import os
import gym
import sys
import random
import itertools
import json
import io
import time as t
from time import time
from copy import copy
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
        # self.path = os.path.join(directory, 'tree.json')
        self.children = []
        self.explored_children = 0
        self.visits = 0
        self.reward = 0
        self.value = 0
        self.state = ""
        # with io.open(self.path, 'a') as f:
        #     json.dump({
        #         'id': self.id,
        #         'parent': -1 if not self.parent else self.parent.id,
        #         'action': self.action,
        #     }, f)

    def to_json(self):
        ret_dict = {}
        ret_dict['id'] = self.id
        # ret_dict['parent'] = self.parent
        ret_dict['action'] = self.action
        # ret_dict['explored_children'] = self.explored_children
        ret_dict['visits'] = self.visits
        ret_dict['value'] = self.value
        ret_dict['reward'] = self.reward
        ret_dict['state'] = self.state
        ret_dict['children'] = [None] * 4
        for child in self.children:
            ret_dict['children'][child.action] = child.to_json()
        return ret_dict


class Runner:
    def __init__(self, rec_dir, env_name, loops=300, max_depth=1000, playouts=10000):
        self.env_name = env_name
        self.dir = rec_dir+'/'+env_name
        os.makedirs(self.dir)

        self.loops = loops
        self.max_depth = max_depth
        self.playouts = playouts

    def print_stats(self, loop, score, avg_time):
        sys.stdout.write('\r%3d   score:%10.3f   avg_time:%4.1f s' % (loop, score, avg_time))
        sys.stdout.flush()

    def run(self):
        best_rewards = []
        start_time = time()
        env = gym.make(self.env_name)
        #env = gym.wrappers.Monitor(env, self.dir)
        #env.monitor.start(self.dir)

        print (self.env_name)

        for loop in range(self.loops):
            env.reset()
            root = Node(None, None, self.dir)

            best_actions = []
            best_reward = float("-inf")

            for _ in range(self.playouts):
                state = copy(env)
                state = gym.make(self.env_name)
                state.reset()
                # del state._monitor

                sum_reward = 0
                node = root
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
                    _, reward, terminal, info = state.step(node.action)
                    node.state = str(info['hkl'][0]) + " " + str(info['hkl'][1]) + " " + str(info['hkl'][2])
                    node.reward = reward
                    sum_reward += reward
                    actions.append(node.action)

                # expansion
                if not terminal:
                    node.children = [Node(node, a, self.dir) for a in combinations(state.action_space)]
                    random.shuffle(node.children)

                # playout
                while not terminal:
                    action = state.action_space.sample()
                    _, reward, terminal, _ = state.step(action)
                    sum_reward += reward
                    actions.append(action)

                    if len(actions) > self.max_depth:
                        sum_reward -= 100
                        break

                # remember best
                if best_reward < sum_reward:
                    best_reward = sum_reward
                    best_actions = actions

                # backpropagate
                while node:
                    node.visits += 1
                    node.value += sum_reward
                    node = node.parent

                # fix monitors not being garbage collected
                #del state._monitor

            sum_reward = 0
            max_reward = 0
            print("best:", end =" ")
            for action in best_actions:
                print(action, end =", ")
                _, reward, terminal, info = env.step(action)
                if reward != 0 and reward > max_reward:
                    info_best = info
                    max_reward = reward
                sum_reward += reward
                if terminal:
                    print("")
                    break
            print("Best HKL: ", str(info_best['hkl'][0]) + " " + str(info_best['hkl'][1]) + " "
                  + str(info_best['hkl'][2]), "With the score of: ", max_reward, "!!!!")

            with open(os.path.join(self.dir, 'tree.json'), 'w') as json_out_file:
                json.dump(root.to_json(), json_out_file)

            best_rewards.append(sum_reward)
            score = max(moving_average(best_rewards, 100))
            avg_time = (time()-start_time)/(loop+1)
            self.print_stats(loop+1, score, avg_time)
        #env.monitor.close()


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

    # Toy text
    Runner(rec_dir, 'hkl-v1',   loops=1, playouts=22*22*22, max_depth=10000).run()
    #Runner(rec_dir, 'NChain-v0', loops=100, playouts=3000, max_depth=50).run()
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
