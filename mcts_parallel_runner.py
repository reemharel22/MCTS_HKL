# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 17:45:36 2018

@author: initial-h
"""
'''
write a root parallel mcts and vote a move like ensemble way
why i do this:
when i train the network, i should try some parameter settings 
and train for a while to compare which is better,
so there are many semi-finished model get useless,it's waste of computation resource 
even though i can continue to train based on the semi-finished model. 
i write the parallel way using MPI,
so that each rank can load different model and then vote the next move to play, besides, 
you can also weights each model to get the weighted next move(i don't do it here but it's easy to realize).

and also each rank can load the same model and vote the next move, 
besides the upper benifit ,it can also improve the strength and save the playout time by parallel.
some other parallel ways can find in《Parallel Monte-Carlo Tree Search》.
'''

import os
import gym
import baselines
from mcts_openai import Runner as MCTSPlayer
from mpi4py import MPI
from collections import Counter

# how  to run (with SLURM):
# salloc -n {n} -p {parition} mpirun python mcts__parallel_runner.py

#MPI setting
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# get rec_dir
if rank == 0 and not os.path.exists('rec'):
    os.makedirs('rec')
    os.makedirs('rec/0')

comm.Barrier()
next_dir = max([int(f) for f in os.listdir('rec')+["0"] if f.isdigit()])+1
rec_dir = 'rec/'+str(next_dir)+'/rank_'+str(rank)
os.makedirs(rec_dir)

player = MCTSPlayer(rec_dir, 'hkl-v1', max_depth=20, playouts=5, verbose=(rank == 0) )# player2 = MCTS_Pure(5,200)

def start_tree_search(start_player=0):
    bcast_move = -1

    # init player
    player.reset_player()

    restart = 0
    end = False

    while True:

        if not end:
            # reset the search tree
            player.reset_player()
            gather_move, move_probs = player.get_action(player.env)

            gather_move_list = comm.gather(gather_move, root=0)

            if rank == 0:
                # gather ecah rank's move and get the most selected one
                print('Ranks recommended actions are ', gather_move_list)
                bcast_move = Counter(gather_move_list).most_common()[0][0]
                # print(board.move_to_location(bcast_move))
                print('Chosen action is ', bcast_move)

        if not end and not restart:
            # bcast the move to other ranks
            bcast_move = comm.bcast(bcast_move, root=0)
            _, reward, end, info = player.env.step(bcast_move)
            if rank == 0:
                print('Current HKL is (', str(info['hkl'][0]) + " " + str(info['hkl'][1]) + " " + str(info['hkl'][2])+')')
                print('The reward for this action was ')
                print(reward)

            # check if game end
            if end:
                if rank == 0:
                    print("Search reached terminal state")
                    print("HKL: ", str(info['hkl'][0]) + " " + str(info['hkl'][1]) + " " + str(info['hkl'][2]))
                    print("Reward is: ", reward)

        else:
            #Note - restart is not implemented yet
            if restart:
                if restart == 'exit':
                    exit()
                player.reset_player()
                restart = 0
                end = False


if __name__ == '__main__':
    start_tree_search()


