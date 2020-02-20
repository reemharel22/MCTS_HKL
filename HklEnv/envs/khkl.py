import os
from os import path
from copy import copy
import random as rand
import pickle
import itertools
import math

import numpy as np
import matplotlib as mpl
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
        n_actions = 4
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
        self.action_space = Discrete(n_actions)

        print("ACTION SPACE INIT IS: ", self.action_space)
        #Graphing and logging arrays
        self.storspot = STOREPATH
        self.rewards = []
        self.hkls = []
        self.zs = []
        self.hs = [1, 0]
        self.ks = [1, 0]
        self.ls = [1, 0]
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

    def step(self, actions=None):
        hkl_prev = [None] * 3
        hkl_prev[0] = self.hs[-1]
        hkl_prev[1] = self.ks[-1]
        hkl_prev[2] = self.ls[-1]
        next_ref = -1
        self.steps += 1
        terminal = False
        if actions == 0:
            reward = 0
        else:
            hkl_prev[actions-1] = hkl_prev[actions-1] + 1
            self.hs[-1] = hkl_prev[0]
            self.ks[-1] = hkl_prev[1]
            self.ls[-1] = hkl_prev[2]
            for i in range(len(self.refList)):
                ref = self.refList[i]
                if self.hs[-1] == ref.hkl[0] and self.ks[-1] == ref.hkl[1] \
                        and self.ls[-1] == ref.hkl[2] :
                   next_ref = i
                   break
        if next_ref == -1 or next_ref == 0:
            reward = 0
        else:
            print("Action: ", actions)

            actions = next_ref
            reward = 0
            print ("Selected HKL: ", self.refList[actions].hkl)

            chisq = None
            dz = None

            self.visited = [self.refList[0]]
            self.visited.append(self.refList[int(actions)])
            self.state[int(actions)] = 1

            #Find the data for this hkl value and add it to the model
            self.model.refList = H.ReflectionList(self.visited)
            # print (self.model.refList)
            self.model._set_reflections()

            self.model.error = [self.error[0]]
            self.model.error.append(self.error[int(actions)])
            self.model.tt = [self.tt[0]]
            self.model.tt = np.append(self.model.tt, [self.tt[int(actions)]])

            self.observed = [self.sfs2[0]]
            self.observed.append(self.sfs2[int(actions)])
            self.model._set_observations(self.observed)
            self.model.update()
            self.hkls = self.refList[0].hkl
            self.hkls.append(self.refList[int(actions)].hkl)
            self.zs = [self.model.atomListModel.atomModels[0].z.value]

            if len(self.visited) > 1:
                # print("about to fit")

                x, dx, chisq, params = better_bumps(self.model)

                dz = params[0].dx
                # Reward function
                # if chisq < 10:
                #     reward += 1000
                # if dz < 2e-3:
                reward += 1/dz

                self.prevChisq = chisq

                self.chisqds.append(chisq)

            self.totReward += reward

            # appened hkls to be logged
            # aaa = self.hkl_to_action(self.hs[-1], self.ks[-1], self.ls[-1])
            print("Reward: ", reward)
        #Ending conditions
        #if (self.prevChisq != None and len(self.visited) > 10 and chisq < 0.05):
        if self.steps > 22*22:
            terminal = True
            self.log()
        else:
            terminal = False

        return self.state, reward, terminal, {'hkl': [self.hs[-1], self.ks[-1], self.ls[-1]]}

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
        self.hs = [1, 0]
        self.ks = [1, 0]
        self.ls = [1, 0]
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
