import os
from os import path
from copy import copy
import random as rand
import pickle
import itertools
import math

import numpy as np
import matplotlib as mpl
#mpl.use('TkAgg')
mpl.use('Agg')
import matplotlib.pyplot as plt

import gym
from gym.utils import seeding
from gym.spaces import Discrete
from baselines.spaces import Bin_Discrete

import hklgen
from hklgen import fswig_hklgen as H
from hklgen import hkl_model as Mod
# our self.model in:
from hklgen import sxtal_model as S

from HklEnv.envs.test_bumps_refl import better_bumps

DATAPATH = os.environ.get('HKL_DATAPATH', None)
if DATAPATH is None:
    DATAPATH = os.path.join(os.path.abspath(os.path.dirname(hklgen.__file__)),
                            'examples', 'sxtal')

STOREPATH = os.environ.get('HKL_STOREPATH', None)
if STOREPATH is None:
    STOREPATH = "."

import bumps.names as bumps
import bumps.fitters as fitters

import numpy as np

# from .find_min import findmin

"""
Re'em notes:
1. The S model is a file called /home/reemh/RL/baselines/pycrysfml/hklgen/sxtal_model.py
2. The ref list is a mapping between the action the agent chooses to the hkl
"""


class HklEnv(gym.Env):

    def __init__(self, reward_scale=100):
        self.reward_scale=reward_scale
        print("Loading problem from %r. Set HklEnv.hkl.DATAPATH"
              " or os.environ['HKL_DATAPATH'] to override." % DATAPATH)
        observedFile = os.path.join(DATAPATH,r"prnio.int")
        infoFile = os.path.join(DATAPATH,r"prnio.cfl")
        self.data_file = observedFile
        #Read data
        self.spaceGroup, self.crystalCell, self.atomList = H.readInfo(infoFile)

        #Return wavelength, refList, sfs2, error, two-theta, and four-circle parameters
        wavelength, refList, sfs2, error = S.readIntFile(observedFile, kind="int", cell=self.crystalCell)
        # print ('Observer file:', observedFile)
        # print ('Atom List:', self.atomList)
        # print ("refList from file:", refList)
        self.wavelength = wavelength
        self.refList = np.array(refList)
        self.sfs2 = sfs2
        self.error = error
        self.tt = [H.twoTheta(H.calcS(self.crystalCell, ref.hkl), wavelength) for ref in refList]
        self.backg = None
        self.exclusions = []
        
        #Set up action space and observation space (in forked baselines)
        self.observation_space = Bin_Discrete(len(self.refList))
        self.action_space = Bin_Discrete(len(self.refList))

        print("ACTION SPACE INIT IS: ", self.action_space)
        #Graphing and logging arrays
        self.storspot = STOREPATH
        self.rewards = []
        self.hkls = []
        self.zs = []
        self.hs = []
        self.ks = []
        self.ls = []
        self.chisqds = []
        self.totReward = 0
        self.envRank = 0
        
        #Setting up boolean masking
        self.valid_actions = np.ones(shape=(5, len(self.refList)))
        self.remaining_acs = np.zeros(198)
        for i in range (0, 198):
            self.remaining_acs[i] = i

        self.episodeNum = 0
        self.steps = 0
        self.prevChisq = None
        
        self.reset()

    def mdts_reward(self, s):

        s = ' '.join(str(x) for x in s)
        s = s.split(' ')
        # Finding the hkl

        for i in range(len(s)):
            if s[i] == "0":
                h = i + 1
            elif s[i] == "1":
                k = i + 1
            elif s[i] == "2":
                l = i + 1

        # Mapping masked action indices to expected indices
        small_scale_ac = actions
        actions = self.hkl_to_action(h, k, l) #self.remaining_acs[int(actions)]
        self.remaining_acs = np.delete(self.remaining_acs, small_scale_ac)
        # print ("actions are:", self.remaining_acs )
        self.steps += 1
        reward = -self.reward_scale

        chisq = None
        dz = None

        self.visited.append(self.refList[int(actions)])
        self.state[int(actions)] = 1

        # Find the data for this hkl value and add it to the model
        self.model.refList = H.ReflectionList(self.visited)
        self.model._set_reflections()

        self.model.error.append(self.error[int(actions)])
        self.model.tt = np.append(self.model.tt, [self.tt[int(actions)]])

        self.observed.append(self.sfs2[int(actions)])
        self.model._set_observations(self.observed)
        self.model.update()
        self.hkls.append(self.refList[int(actions)].hkl)
        self.zs.append(self.model.atomListModel.atomModels[0].z.value)

        if len(self.visited) > 1:
            # print("about to fit")

            x, dx, chisq, params = better_bumps(self.model)

            dz = params[0].dx

            # Reward function
            if chisq < 10:
                reward += 1000
                if dz < 2e-3:
                    reward += 1 / dz

            self.prevChisq = chisq

            self.chisqds.append(chisq)

        self.totReward += reward

        # appened hkls to be logged
        hkls_arr = np.asarray(self.refList[int(actions)].hkl)
        self.hs.append(hkls_arr[0])
        self.ks.append(hkls_arr[1])
        self.ls.append(hkls_arr[2])

        # Provide an array of valid actions for boolean mask
        for i in range(len(self.valid_actions)):
            self.valid_actions[i] = (self.state + 1) % 2

        # Ending conditions
        if (self.prevChisq != None and len(self.visited) > 10 and chisq < 0.05):
            terminal = True
            self.log()
        if (self.steps > 30):
            terminal = True
            self.log()
        else:
            terminal = False

        return reward

    def hkl_to_action(self, h, k, l):
        for i in range (len(self.refList)):
            h_ref = self.refList[i].hkl[0]
            k_ref = self.refList[i].hkl[1]
            l_ref = self.refList[i].hkl[2]
            if h_ref == h and k_ref == k and l_ref == l:
                print ("Found hkl")
                return i
        return -1

    def get_hkl(self, s):
        for i in range(len(s)):
            if s[i] == "0":
                h = i + 1
            elif s[i] == "1":
                k = i + 1
            elif s[i] == "2":
                l = i + 1
        print( h % 11, k % 11, l %11)
        return h % 11, k % 11, l %11

    def get_id(self, arr):
        def nCr(n, r):
            if n < 0 or r < 0 or r > n:
                return 0
            f = math.factorial
            return f(n) / f(r) / f(n - r)

        n = len(arr)
        r = sum(arr)
        id = 0

        for i in range(n):
            id += arr[i] * nCr(n - 1, r)
            r -= arr[i]
            n -= 1
        return int(id + 1)

    def get_hkl_from_file_id(self, arr_bit):
        line_num = self.get_id(arr_bit)
        if line_num > 192:
            return -1,-1,-1
        print (line_num)
        f1 = open(self.data_file, "r")
        lines = f1.readlines()
        line = lines[line_num + 2].split()
        h = int(line[0])
        k = int(line[1])
        l = int(line[2])
        return h, k, l

    def step(self, actions=None):

        print("\n\nstepping", actions)
        #Mapping masked action indices to expected indices

        small_scale_ac = actions
        # actions = self.remaining_acs[int(actions)]
        print ("Actions:", actions)

        # self.remaining_acs = np.delete(self.remaining_acs, small_scale_ac)
        #self.action_space = np.delete(self.action_space, small_scale_ac)
        # print ("actions are:", self.remaining_acs )
        self.steps += 1
        reward = -self.reward_scale

        chisq = None
        dz = None

        self.visited.append(self.refList[int(actions)])
        self.state[int(actions)] = 1
        self.action_space.update(actions)

        #Find the data for this hkl value and add it to the model
        self.model.refList = H.ReflectionList(self.visited)
        print (self.model.refList)
        self.model._set_reflections()

        self.model.error.append(self.error[int(actions)])
        self.model.tt = np.append(self.model.tt, [self.tt[int(actions)]])

        self.observed.append(self.sfs2[int(actions)])
        self.model._set_observations(self.observed)
        self.model.update()
        self.hkls.append(self.refList[int(actions)].hkl)
        self.zs.append(self.model.atomListModel.atomModels[0].z.value)

        if len(self.visited) > 1:
            # print("about to fit")

            x, dx, chisq, params = better_bumps(self.model)

            dz = params[0].dx

            # Reward function
            if chisq < 10:
                reward += 1000
                if dz < 2e-3:
                    reward += 1/dz

            self.prevChisq = chisq

            self.chisqds.append(chisq)

        self.totReward += reward

        # appened hkls to be logged
        hkls_arr = np.asarray(self.refList[int(actions)].hkl)
        self.hs.append(hkls_arr[0])
        self.ks.append(hkls_arr[1])
        self.ls.append(hkls_arr[2])
        # aaa = self.hkl_to_action(self.hs[-1], self.ks[-1], self.ls[-1])

        #Provide an array of valid actions for boolean mask
        for i in range(len(self.valid_actions)):
            self.valid_actions[i] = (self.state +1) % 2

        #Ending conditions
        #if (self.prevChisq != None and len(self.visited) > 10 and chisq < 0.05):
        if (len(self.visited) > 30):
            terminal = True
            self.log()
        #if (self.steps > 30):
        #    terminal = True
        #    self.log()
        else:
            terminal = False


        # print ("########################################### REEEEEEEE######################")
    # print (self.state, reward, terminal,  self.valid_actions)
        return self.state, reward, terminal, {'valid_actions': self.valid_actions}

    def reset(self):
        #Make a cell
        cell = Mod.makeCell(self.crystalCell, self.spaceGroup.xtalSystem)

        #Define a model
        self.model = S.Model([], [], self.backg, self.wavelength, self.spaceGroup, cell,
                             self.atomList, self.exclusions,
                    scale=0.06298, error=[],  extinction=[0.0001054])

        #Set a range on the x value of the first atom in the model
        self.model.atomListModel.atomModels[0].z.value = 0.25
        self.model.atomListModel.atomModels[0].z.range(0,0.5)

        self.visited = []
        self.observed = []
        self.hkls = []
        self.zs = []
        self.hs = []
        self.ks = []
        self.ls = []
        self.chisqds = []
        self.totReward = 0
        
        #Resetting boolean mask mapping
        self.valid_actions = np.ones(shape = (5, len(self.refList)))
        self.remainingActions = []
        
        #TODO remove hardcoded 
        self.remaining_acs = np.zeros(198)
        for i in range (0, 198):
            self.remaining_acs[i] = i
        
        for i in range(len(self.refList)):
            self.remainingActions.append(i)

        self.prevChisq = None
        self.steps = 0

        self.state = np.zeros(len(self.refList))

        return self.state

    def giveRank(self, subrank):
        self.envRank = subrank
        
    def log(self):
        self.episodeNum += 1

        file = open(self.storspot +"/hklLog-" + str(self.episodeNum) + "_" + str(self.envRank) + ".txt", "w+")
        file.write(str(self.hkls))
        file.close()
        
        filename = self.storspot + "/zLog-" + str(self.episodeNum) + "_" + str(self.envRank) + ".txt"
        np.savetxt(filename, self.zs)
        
        # filename = self.storspot +"/hLog-" + str(self.episodeNum) + "_" + str(self.envRank) + ".txt"
        # np.savetxt(filename, self.hs)
        
        # filename = self.storspot +"/kLog-" + str(self.episodeNum) + "_" + str(self.envRank) + ".txt"
        # np.savetxt(filename, self.ks)
        
        filename = self.storspot +"/lLog-" + str(self.episodeNum) + "_" + str(self.envRank) + ".txt"
        np.savetxt(filename, self.ls)
        
        filename = self.storspot +"/chiLog-" + str(self.episodeNum) + "_" + str(self.envRank) + ".txt"
        np.savetxt(filename, self.chisqds)
        
        self.rewards.append(self.totReward)
        filename = self.storspot +"/rewardLog-" +  str(self.envRank) + ".txt"
        np.savetxt(filename, self.rewards)   
        
        print("ENDED EPISODE")

    @property
    def states(self):
        return dict(shape=self.state.shape, type='float')

    @property
    def actions(self):
        return dict(num_actions=len(self.refList), type='int')

    def clone_full_state(self):
        return self.state

    def restore_full_state(self, state):
        self.state = state
        for action, isselected in enumerate(state):
            if isselected == 1:
                self.step(action)
                


def profile(fn, *args, **kw):
    """
    Profile a function called with the given arguments.
    """
    import cProfile
    import pstats

    print("in profile", fn, args, kw)
    result = [None]
    def call():
        try:
            result[0] = fn(*args, **kw)
        except BaseException as exc:
            result.append(exc)
    datafile = 'profile.out'
    cProfile.runctx('call()', dict(call=call), {}, datafile)
    stats = pstats.Stats(datafile)
    order = 'cumulative'
    stats.sort_stats(order)
    stats.print_stats()
    os.unlink(datafile)
    if len(result) > 1:
        raise result[1]
    return result[0]
    
class Profiler(object):
    def __init__(self,  fn, datafile='profile.out'):
        self.fn = fn
        self.datafile = datafile
        self.first = True
        
    def __call__(self, *args, **kw):
        if self.first:
            self.first = False
            import cProfile
            
            result = [None]
            def call():
                result[0] = self.fn(fn, *args, **kw)
            
            cProfile.runctx('call()', dict(call=call), {}, self.datafile)
            self.summarize()
            return result[0]
        else:
            return self.fn(*args, **kw)
            
    def summarize(self):
        """
        Profile a function called with the given arguments.
        """
        import pstats, sys

        with open("stats.out", "w") as stream:
            stats = pstats.Stats(self.datafile, stream=stream)
            order = 'cumulative'
            stats.sort_stats(order)
            stats.print_stats()
        
    def cleanup():
        self.summarize()
        os.unlink(self.datafile)
