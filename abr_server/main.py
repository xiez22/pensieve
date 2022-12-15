import numpy as np
import fixed_env as env
import load_trace
from models.ABR import *
from models.pensieve.pensieve_abr import Pensieve
from constant import *
from parse import parse_arg
import os
import time as tm

opt=parse_arg()

def main():
    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace()

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    if os.path.exists(LOG_FILE + opt.abr)==0: os.mkdir(LOG_FILE + opt.abr)
    log_path = LOG_FILE + opt.abr + '/' + all_file_names[net_env.trace_idx]
    print(log_path)
    log_file = open(log_path, 'w')

    epoch = 0
    time_stamp = 0
    
    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY
    
    reward_sum=[]
    call_time_sum =[]
    all_reward_sum=[]
    all_time_sum=[]
    trace_count = 0
    
    if opt.abr=='bb': abr=BB()
    elif opt.abr=='rb': abr=RB()
    elif opt.abr=='mpc': abr=MPC()
    elif opt.abr == 'fastmpc': abr=FastMPC()
    elif opt.abr == 'festive': abr=Festive()
    elif opt.abr == 'pensieve': abr=Pensieve()
    else:
        raise NotImplementedError(f"Unrecognized ABR algo: {opt.abr}.")
    
    abr.initial()

    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                           VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
        last_reward=reward
        last_bit_rate = bit_rate

        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write('{:.2f} \t {:5d} \t {:.2f} \t {:.2f} \t {:5d} \t {:.2f} \t {:.2f}\n'.format(
            time_stamp / M_IN_K,VIDEO_BIT_RATE[bit_rate],
            buffer_size,rebuf,video_chunk_size,
            delay,reward))
        log_file.flush()
        reward_sum.append(reward)
        
        timestamp_start = tm.time()
        bit_rate=abr.run(delay, sleep_time, buffer_size, rebuf, \
                         video_chunk_size, next_video_chunk_sizes, \
                         end_of_video, video_chunk_remain,last_bit_rate,last_reward)
        timestamp_end = tm.time()
        call_time_sum.append(timestamp_end - timestamp_start)
        if end_of_video:
            log_file.write('Total Chunks is '+str(len(reward_sum))+'  Average Reward is '+str(np.mean(reward_sum))+
                           '  Running Time is '+str(np.mean(call_time_sum)))
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here
            all_reward_sum.append(np.mean(reward_sum))
            all_time_sum.append(np.mean(call_time_sum))
            reward_sum=[]
            call_time_sum=[]
            abr.initial()
            
            trace_count += 1
            if trace_count%10==0 or trace_count== len(all_file_names): print("trace count", trace_count)
            if trace_count >= len(all_file_names):
                log_path = LOG_FILE + opt.abr + '/ALL'
                log_file = open(log_path, 'w')
                for i in range (len(all_reward_sum)):
                    log_file.write('Traces '+ str(i+1) +': Reward is '+str(all_reward_sum[i])+
                           '  Running Time is '+str(np.mean(all_time_sum[i]))+'\n')
                log_file.write('Over All Traces, Average Reward is '+str(np.mean(all_reward_sum))+
                           '  Running Time is '+str(np.mean(all_time_sum)))
                log_file.close()
                break
            log_path = LOG_FILE + opt.abr + '/' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')

if __name__ == '__main__':
    main()
