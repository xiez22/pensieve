import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import copy
import itertools
from constant import *

class BB:
    def __init__(self):
     # fill your self params
        self.buffer_size = 0

     # Intial
    def initial(self):
     # Initail your session or something

     # restore neural net parameters
        self.buffer_size = 0

     #Define your al
    def run(self, delay, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        video_chunk_remain, last_bitrate, last_reward):
        
        if buffer_size < RESEVOIR:
            bit_rate = 0
        elif buffer_size >= RESEVOIR + CUSHION:
            bit_rate = A_DIM - 1
        else:
            bit_rate = (A_DIM - 1) * (buffer_size - RESEVOIR) / float(CUSHION)

        bit_rate = int(bit_rate)
        return bit_rate


class RB:
    def __init__(self):
        # fill your self params
        self.buffer_size = 0

    # Intial
    def initial(self):
        # Initail your session or something
        self.s_batch = [np.zeros((S_INFO, S_LEN))]

        # restore neural net parameters
        self.p_rb = 1.0

    #Define your al
    def run(self, delay, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        video_chunk_remain, last_bitrate, last_reward):
        # retrieve previous state
        if len(self.s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(self.s_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
        try:
            state[0, -1] = VIDEO_BIT_RATE[last_bitrate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
            state[2, -1] = rebuf / M_IN_K
            state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[4, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        except ZeroDivisionError:
            # this should occur VERY rarely (1 out of 3000), should be a dash issue
            # in this case we ignore the observation and roll back to an eariler one
            if len(self.s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(self.s_batch[-1], copy=True)

        # first get harmonic mean of last 5 bandwidths
        past_bandwidths = state[3,-5:]
        while past_bandwidths[0] == 0.0:
            past_bandwidths = past_bandwidths[1:]

        # Predict bandwidths
        bandwidth_sum = 0
        for past_val in past_bandwidths:
            bandwidth_sum += (1 / float(past_val))
        future_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidths))
        tmp_bit_rate = future_bandwidth * self.p_rb * 1_000 * 8

        # Select bit_rate
        bit_rate = BITRATE_LEVELS - 1
        for cur_bit_rate in range(BITRATE_LEVELS - 1, -1, -1):
            if tmp_bit_rate >= VIDEO_BIT_RATE[cur_bit_rate]:
                bit_rate = cur_bit_rate
                break
        else:
            bit_rate = 0

        self.s_batch.append(state)
        return bit_rate


class Festive:
    def __init__(self):
        # fill your self params
        self.buffer_size = 0

    # Intial
    def initial(self):
        # Initail your session or something
        self.s_batch = [np.zeros((S_INFO, S_LEN))]
        self.p = 0.85
        self.switch_up_count = 0
        self.quality_log = []
        self.alpha = 12

    def _get_stability_score(self, b: int, b_ref: int, b_cur: int):
        n = 0
        for i in range(len(self.quality_log) - 1):
            if self.quality_log[i] != self.quality_log[i + 1]:
                n += 1
        if b != b_cur:
            n += 1
        return 2 ** n

    def _get_efficiency_score(self, b, b_ref, w):
        return abs(VIDEO_BIT_RATE[b] / min(w, VIDEO_BIT_RATE[b_ref]) - 1)

    def _get_combined_score(self, b, b_ref, b_cur, w):
        stability_score = self._get_stability_score(b, b_ref, b_cur)
        efficiency_socre = self._get_efficiency_score(b, b_ref, w)
        total_score = stability_score + self.alpha * efficiency_socre
        return total_score

    #Define your al
    def run(self, delay, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        video_chunk_remain, last_bitrate, last_reward):
        # retrieve previous state
        if len(self.s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(self.s_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
        try:
            state[0, -1] = VIDEO_BIT_RATE[last_bitrate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
            state[2, -1] = rebuf / M_IN_K
            state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[4, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        except ZeroDivisionError:
            # this should occur VERY rarely (1 out of 3000), should be a dash issue
            # in this case we ignore the observation and roll back to an eariler one
            if len(self.s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(self.s_batch[-1], copy=True)
        self.s_batch.append(state)

        # first get harmonic mean of last 5 bandwidths
        past_bandwidths = state[3,-5:]
        while past_bandwidths[0] == 0.0:
            past_bandwidths = past_bandwidths[1:]

        # Predict bandwidths
        bandwidth_sum = 0
        for past_val in past_bandwidths:
            bandwidth_sum += (1 / float(past_val))
        future_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidths)) * 1_000 * 8

        # ===================== FESTIVE Logic ============================
        self.quality_log.append(last_bitrate)
        tmp_bit_rate = future_bandwidth * self.p

        # Select bit_rate
        b_target = BITRATE_LEVELS - 1
        for cur_bit_rate in range(BITRATE_LEVELS - 1, -1, -1):
            if tmp_bit_rate >= VIDEO_BIT_RATE[cur_bit_rate]:
                b_target = cur_bit_rate
                break
        else:
            b_target = 0

        # Compute b_ref
        b_cur = last_bitrate
        b_ref = 0
        if b_target > b_cur:
            self.switch_up_count += 1
            if self.switch_up_count > b_cur:
                b_ref = b_cur + 1
            else:
                b_ref = b_cur
        elif b_target < b_cur:
            b_ref = b_cur - 1
            self.switch_up_count = 0
        else:
            b_ref = b_cur
            self.switch_up_count = 0

        # Delay update
        if b_ref != b_cur:
            # Need to switch
            # Compute score
            score_cur = self._get_combined_score(b_cur, b_ref, b_cur, future_bandwidth)
            score_ref = self._get_combined_score(b_ref, b_ref, b_cur, future_bandwidth)

            if score_cur <= score_ref:
                bitrate = b_cur
            else:
                bitrate = b_ref
            if bitrate > b_cur:
                self.switch_up_count = 0
        else:
            bitrate = b_cur

        return bitrate


class MPC:
    def __init__(self):
        self.CHUNK_COMBO_OPTIONS = []
        # past errors in bandwidth
        self.past_errors = []
        self.past_bandwidth_ests = []
        self.video_size = {}  # in bytes
        for bitrate in range(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

        # make chunk combination options
        for combo in itertools.product([0,1,2,3,4,5], repeat=5):
            self.CHUNK_COMBO_OPTIONS.append(combo)

     # Intial
    def initial(self):
        self.action_vec = np.zeros(A_DIM)
        self.action_vec[DEFAULT_QUALITY] = 1

        self.s_batch = [np.zeros((S_INFO, S_LEN))]
        self.a_batch = [self.action_vec]
        self.r_batch = []
        self.entropy_record = []
        
    def get_chunk_size(self, chunk_quality, index):
        if ( index < 0 or index >= TOTAL_VIDEO_CHUNK ): return 0
        else: return self.video_size[chunk_quality][index]

     #Define your al
    def run(self, delay, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        video_chunk_remain, last_bitrate, last_reward):
        
        self.r_batch.append(last_reward)
        # retrieve previous state
        if len(self.s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(self.s_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
        try:
            state[0, -1] = VIDEO_BIT_RATE[last_bitrate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
            state[2, -1] = rebuf / M_IN_K
            state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[4, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        except ZeroDivisionError:
            # this should occur VERY rarely (1 out of 3000), should be a dash issue
            # in this case we ignore the observation and roll back to an eariler one
            if len(self.s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(self.s_batch[-1], copy=True)
        # state[5: 10, :] = future_chunk_sizes / M_IN_K / M_IN_K

        # ================== MPC =========================
        curr_error = 0 # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
        if ( len(self.past_bandwidth_ests) > 0 ):
            curr_error  = abs(self.past_bandwidth_ests[-1]-state[3,-1])/float(state[3,-1])
        self.past_errors.append(curr_error)

        # pick bitrate according to MPC           
        # first get harmonic mean of last 5 bandwidths
        past_bandwidths = state[3,-5:]
        while past_bandwidths[0] == 0.0:
            past_bandwidths = past_bandwidths[1:]
        #if ( len(state) < 5 ):
        #    past_bandwidths = state[3,-len(state):]
        #else:
        #    past_bandwidths = state[3,-5:]
        bandwidth_sum = 0
        for past_val in past_bandwidths:
            bandwidth_sum += (1/float(past_val))
        harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))

        # future bandwidth prediction
        # divide by 1 + max of last 5 (or up to 5) errors
        max_error = 0
        error_pos = -5
        if ( len(self.past_errors) < 5 ):
            error_pos = -len(self.past_errors)
        max_error = float(max(self.past_errors[error_pos:]))
        future_bandwidth = harmonic_bandwidth/(1+max_error)  # robustMPC here
        self.past_bandwidth_ests.append(harmonic_bandwidth)


        # future chunks length (try 4 if that many remaining)
        last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain)
        future_chunk_length = MPC_FUTURE_CHUNK_COUNT
        if ( TOTAL_VIDEO_CHUNK - last_index < 5 ):
            future_chunk_length = TOTAL_VIDEO_CHUNK - last_index

        # all possible combinations of 5 chunk bitrates (9^5 options)
        # iterate over list and for each, compute reward and store max reward combination
        max_reward = -100000000
        best_combo = ()
        start_buffer = buffer_size
        #start = time.time()
        for full_combo in self.CHUNK_COMBO_OPTIONS:
            combo = full_combo[0:future_chunk_length]
            # calculate total rebuffer time for this combination (start with start_buffer and subtract
            # each download time and add 2 seconds in that order)
            curr_rebuffer_time = 0
            curr_buffer = start_buffer
            bitrate_sum = 0
            smoothness_diffs = 0
            last_quality = int( last_bitrate )
            for position in range(0, len(combo)):
                chunk_quality = combo[position]
                index = last_index + position + 1 # e.g., if last chunk is 3, then first iter is 3+0+1=4
                download_time = (self.get_chunk_size(chunk_quality, index)/1000000.)/future_bandwidth # this is MB/MB/s --> seconds
                if ( curr_buffer < download_time ):
                    curr_rebuffer_time += (download_time - curr_buffer)
                    curr_buffer = 0
                else:
                    curr_buffer -= download_time
                curr_buffer += 4
                bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
                smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
                # bitrate_sum += BITRATE_REWARD[chunk_quality]
                # smoothness_diffs += abs(BITRATE_REWARD[chunk_quality] - BITRATE_REWARD[last_quality])
                last_quality = chunk_quality
            # compute reward for this combination (one reward per 5-chunk combo)
            # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s
            
            reward = (bitrate_sum/1000.) - (REBUF_PENALTY*curr_rebuffer_time) - (smoothness_diffs/1000.)
            # reward = bitrate_sum - (8*curr_rebuffer_time) - (smoothness_diffs)


            if ( reward >= max_reward ):
                if (best_combo != ()) and best_combo[0] < combo[0]:
                    best_combo = combo
                else:
                    best_combo = combo
                max_reward = reward
                # send data to html side (first chunk of best combo)
                send_data = 0 # no combo had reward better than -1000000 (ERROR) so send 0
                if ( best_combo != () ): # some combo was good
                    send_data = best_combo[0]

        bit_rate = send_data
        # hack
        # if bit_rate == 1 or bit_rate == 2:
        #    bit_rate = 0

        # ================================================

        # Note: we need to discretize the probability into 1/RAND_RANGE steps,
        # because there is an intrinsic discrepancy in passing single state and batch states

        self.s_batch.append(state)
        return bit_rate


class FastMPC:
    def __init__(self):
        self.CHUNK_COMBO_OPTIONS = []
        # past errors in bandwidth
        self.past_errors = []
        self.past_bandwidth_ests = []
        self.video_size = {}  # in bytes
        for bitrate in range(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

        # make chunk combination options
        for combo in itertools.product([0,1,2,3,4,5], repeat=5):
            self.CHUNK_COMBO_OPTIONS.append(combo)

     # Intial
    def initial(self):
        self.action_vec = np.zeros(A_DIM)
        self.action_vec[DEFAULT_QUALITY] = 1

        self.s_batch = [np.zeros((S_INFO, S_LEN))]
        self.a_batch = [self.action_vec]
        self.r_batch = []
        self.entropy_record = []
        
    def get_chunk_size(self, chunk_quality, index):
        if ( index < 0 or index >= TOTAL_VIDEO_CHUNK ): return 0
        else: return self.video_size[chunk_quality][index]

     #Define your al
    def run(self, delay, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        video_chunk_remain, last_bitrate, last_reward):
        
        self.r_batch.append(last_reward)
        # retrieve previous state
        if len(self.s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(self.s_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
        try:
            state[0, -1] = VIDEO_BIT_RATE[last_bitrate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
            state[2, -1] = rebuf / M_IN_K
            state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[4, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        except ZeroDivisionError:
            # this should occur VERY rarely (1 out of 3000), should be a dash issue
            # in this case we ignore the observation and roll back to an eariler one
            if len(self.s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(self.s_batch[-1], copy=True)
        # state[5: 10, :] = future_chunk_sizes / M_IN_K / M_IN_K

        # ================== FastMPC =========================
        curr_error = 0 # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
        if ( len(self.past_bandwidth_ests) > 0 ):
            curr_error  = abs(self.past_bandwidth_ests[-1]-state[3,-1])/float(state[3,-1])
        self.past_errors.append(curr_error)

        # pick bitrate according to MPC           
        # first get harmonic mean of last 5 bandwidths
        past_bandwidths = state[3,-5:]
        while past_bandwidths[0] == 0.0:
            past_bandwidths = past_bandwidths[1:]
        #if ( len(state) < 5 ):
        #    past_bandwidths = state[3,-len(state):]
        #else:
        #    past_bandwidths = state[3,-5:]
        bandwidth_sum = 0
        for past_val in past_bandwidths:
            bandwidth_sum += (1/float(past_val))
        future_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))

        # future chunks length (try 4 if that many remaining)
        last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain)
        future_chunk_length = MPC_FUTURE_CHUNK_COUNT
        if ( TOTAL_VIDEO_CHUNK - last_index < 5 ):
            future_chunk_length = TOTAL_VIDEO_CHUNK - last_index

        # all possible combinations of 5 chunk bitrates (9^5 options)
        # iterate over list and for each, compute reward and store max reward combination
        max_reward = -100000000
        best_combo = ()
        start_buffer = buffer_size
        #start = time.time()
        for full_combo in self.CHUNK_COMBO_OPTIONS:
            combo = full_combo[0:future_chunk_length]
            # calculate total rebuffer time for this combination (start with start_buffer and subtract
            # each download time and add 2 seconds in that order)
            curr_rebuffer_time = 0
            curr_buffer = start_buffer
            bitrate_sum = 0
            smoothness_diffs = 0
            last_quality = int( last_bitrate )
            for position in range(0, len(combo)):
                chunk_quality = combo[position]
                index = last_index + position + 1 # e.g., if last chunk is 3, then first iter is 3+0+1=4
                download_time = (self.get_chunk_size(chunk_quality, index)/1000000.)/future_bandwidth # this is MB/MB/s --> seconds
                if ( curr_buffer < download_time ):
                    curr_rebuffer_time += (download_time - curr_buffer)
                    curr_buffer = 0
                else:
                    curr_buffer -= download_time
                curr_buffer += 4
                bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
                smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
                # bitrate_sum += BITRATE_REWARD[chunk_quality]
                # smoothness_diffs += abs(BITRATE_REWARD[chunk_quality] - BITRATE_REWARD[last_quality])
                last_quality = chunk_quality
            # compute reward for this combination (one reward per 5-chunk combo)
            # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s
            
            reward = (bitrate_sum/1000.) - (REBUF_PENALTY*curr_rebuffer_time) - (smoothness_diffs/1000.)
            # reward = bitrate_sum - (8*curr_rebuffer_time) - (smoothness_diffs)


            if ( reward >= max_reward ):
                if (best_combo != ()) and best_combo[0] < combo[0]:
                    best_combo = combo
                else:
                    best_combo = combo
                max_reward = reward
                # send data to html side (first chunk of best combo)
                send_data = 0 # no combo had reward better than -1000000 (ERROR) so send 0
                if ( best_combo != () ): # some combo was good
                    send_data = best_combo[0]

        bit_rate = send_data
        # hack
        # if bit_rate == 1 or bit_rate == 2:
        #    bit_rate = 0

        # ================================================

        # Note: we need to discretize the probability into 1/RAND_RANGE steps,
        # because there is an intrinsic discrepancy in passing single state and batch states

        self.s_batch.append(state)
        return bit_rate
