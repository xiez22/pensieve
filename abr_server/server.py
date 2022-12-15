#!/usr/bin/env python
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
import base64
import urllib
import sys
import os
import json
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
from constant import *
from models.ABR import *
from models.pensieve.pensieve_abr import Pensieve

import numpy as np
import time
import itertools


# Define port map: [abr: port]
abr_port_dict = {
    'pensieve': 12300,
    'bb': 12301,
    'rb': 12302,
    'festive': 12303,
    'mpc': 12304,
    'fastmpc': 12305,
    'simple': 12306
}


def make_request_handler(abr_model, input_dict):

    class Request_Handler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.input_dict = input_dict
            self.log_file = input_dict['log_file']
            
            BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length))
            print(post_data)

            if ( 'pastThroughput' in post_data ):
                # @Hongzi: this is just the summary of throughput/quality at the end of the load
                # so we don't want to use this information to send back a new quality
                print("Summary: ", post_data)
            else:
                rebuffer_time = float(post_data['RebufferTime'] -self.input_dict['last_total_rebuf'])

                # --linear reward--
                reward = VIDEO_BIT_RATE[post_data['lastquality']] / M_IN_K \
                        - REBUF_PENALTY * rebuffer_time / M_IN_K \
                        - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[post_data['lastquality']] -
                                                  self.input_dict['last_bit_rate']) / M_IN_K
                # --log reward--
                # log_bit_rate = np.log(VIDEO_BIT_RATE[post_data['lastquality']] / float(VIDEO_BIT_RATE[0]))   
                # log_last_bit_rate = np.log(self.input_dict['last_bit_rate'] / float(VIDEO_BIT_RATE[0]))

                # reward = log_bit_rate \
                #          - 4.3 * rebuffer_time / M_IN_K \
                #          - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

                # --hd reward--
                # reward = BITRATE_REWARD[post_data['lastquality']] \
                #         - 8 * rebuffer_time / M_IN_K - np.abs(BITRATE_REWARD[post_data['lastquality']] - BITRATE_REWARD_MAP[self.input_dict['last_bit_rate']])
                
                self.input_dict['last_bit_rate'] = VIDEO_BIT_RATE[post_data['lastquality']]
                self.input_dict['last_total_rebuf'] = post_data['RebufferTime']
                video_chunk_fetch_time = post_data['lastChunkFinishTime'] - post_data['lastChunkStartTime']
                video_chunk_size = post_data['lastChunkSize']

                # compute number of video chunks left
                video_chunk_remain = TOTAL_VIDEO_CHUNKS - self.input_dict['video_chunk_count']
                self.input_dict['video_chunk_count'] += 1
                
                next_video_chunk_sizes = post_data['nextChunkSize']
                # log wall_time, bit_rate, buffer_size, rebuffer_time, video_chunk_size, download_time, reward
                self.log_file.write('{:.2f} \t {:5d} \t {:.2f} \t {:.2f} \t {:5d} \t {:.2f} \t {:.2f}\n'.format(
                    time.time(),VIDEO_BIT_RATE[post_data['lastquality']],post_data['buffer'],
                    rebuffer_time / M_IN_K,video_chunk_size,video_chunk_fetch_time,reward))
                self.log_file.flush()

                
                
                start = time.time()
                if abr_model is not None:
                    send_data=abr_model.run(video_chunk_fetch_time, post_data['buffer'], rebuffer_time, \
                                video_chunk_size, next_video_chunk_sizes, \
                                video_chunk_remain, post_data['lastquality'], reward)
                                
                    send_data=str(send_data)
                else:
                    send_data = '0'

                end = time.time()
                print("TOOK: " + str(end-start))

                end_of_video = False
                if ( post_data['lastRequest'] == TOTAL_VIDEO_CHUNKS-1 ):
                    send_data = "REFRESH"
                    end_of_video = True
                    self.input_dict['last_total_rebuf'] = 0
                    self.input_dict['last_bit_rate'] = DEFAULT_QUALITY
                    self.input_dict['video_chunk_count'] = 0
                    print("Video ends\n")
                    self.log_file.write('\n')  # so that in the log we know where video ends

                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.send_header('Content-Length', len(send_data))
                self.send_header('Access-Control-Allow-Origin', "*")
                self.end_headers()
                self.wfile.write(send_data.encode())

                # record [state, action, reward]
                # put it here after training, notice there is a shift in reward storage

                if end_of_video:
                    abr_model.initial()

        def do_GET(self):
            print('GOT REQ',sys.stderr)
            self.send_response(200)
            #self.send_header('Cache-Control', 'Cache-Control: no-cache, no-store, must-revalidate max-age=0')
            self.send_header('Cache-Control', 'max-age=3000')
            self.send_header('Content-Length', 20)
            self.end_headers()
            self.wfile.write("console.log('here');")

        def log_message(self, format, *args):
            return

    return Request_Handler


def run(ABR,server_class=HTTPServer, port=8333, log_file_path=LOG_FILE):

    np.random.seed(RANDOM_SEED)

    if not os.path.exists(SUMMARY_DIR):
        os.mkdir(SUMMARY_DIR)

    if ABR=='bb': abr=BB()
    elif ABR=='rb': abr=RB()
    elif ABR=='mpc': abr=MPC()
    elif ABR == 'fastmpc': abr=FastMPC()
    elif ABR == 'festive': abr=Festive()
    elif ABR == 'pensieve': abr=Pensieve()
    elif ABR == 'simple': abr=None
    else:
        raise NotImplementedError(f"Unrecognized ABR algo: {ABR}.")

    with open(log_file_path, 'w') as log_file:
        if abr is not None:
            abr.initial()

        last_bit_rate = DEFAULT_QUALITY
        last_total_rebuf = 0
        # need this storage, because observation only contains total rebuffering time
        # we compute the difference to get

        video_chunk_count = 0

        input_dict = {'log_file': log_file,
                      'last_bit_rate': last_bit_rate,
                      'last_total_rebuf': last_total_rebuf,
                      'video_chunk_count': video_chunk_count}

        # interface to abr_rl server
        handler_class = make_request_handler(abr_model=abr, input_dict=input_dict)

        server_address = ('localhost', port)
        httpd = server_class(server_address, handler_class)
        print('Listening on port ' + str(port))
        httpd.serve_forever()

def main():
    abr=sys.argv[1]
    trace_file = sys.argv[2]
    print(f'Usage: python3 server.py [abr] [trace_name]')
    print(f'ABR: {abr} Port: {abr_port_dict[abr]} Trace: {trace_file}')
    # port=sys.argv[3]
    print(sys.argv)
    if abr not in ['bb','rb','mpc','festive','fastmpc','pensieve', 'simple']: 
        raise NotImplementedError(f"Unrecognized ABR algo: {abr}.")
    if os.path.exists(LOG_FILE + 'server')==0: os.mkdir(LOG_FILE + 'server')
    if os.path.exists(LOG_FILE + 'server/'+abr)==0: os.mkdir(LOG_FILE + 'server/'+abr)
    run(ABR=abr, port=abr_port_dict[abr], log_file_path=LOG_FILE + 'server/' + abr + '/' + trace_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard interrupted.")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
