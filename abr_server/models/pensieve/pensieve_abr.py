import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import copy
import itertools
from models.pensieve.A3C import ActorNetwork
from torch.distributions import Categorical
from constant import *


# Override some params
S_INFO = 6
ACTOR_MODEL = './trained_model/actor.pt'


class Pensieve:
    def __init__(self):
        # fill your self params
        self.buffer_size = 0
        torch.set_num_threads(1)
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

        # Load model
        self.net = ActorNetwork([S_INFO, S_LEN], A_DIM)

        # restore neural net parameters
        try:
            self.net.load_state_dict(torch.load(ACTOR_MODEL))
            print("[Pensieve] Testing model restored.")
        except Exception as err:
            print(err)
            raise FileNotFoundError(f"[Pensieve] Pensieve model: {ACTOR_MODEL} not found. Please run 'python3 -m models.pensieve.train' to train a model first!")

    # Intial
    def initial(self):
        self.state = torch.zeros([S_INFO, S_LEN])

    #Define your al
    def run(self, delay, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        video_chunk_remain, last_bitrate, last_reward):
        # dequeue history record
        self.state = torch.roll(self.state, -1, dims=-1)

        # this should be S_INFO number of terms
        self.state[0, -1] = VIDEO_BIT_RATE[last_bitrate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        self.state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        self.state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        self.state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        self.state[4, :A_DIM] = torch.tensor(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        self.state[5, -1] = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        with torch.no_grad():
            probability = self.net.forward(self.state.unsqueeze(0))
            m = Categorical(probability)
            bit_rate = m.sample().item()

        return bit_rate
