#COOKED_TRACE_FOLDER = './datasets/network_trace/metis_cooked_traces/'
COOKED_TRACE_FOLDER = './datasets/network_trace/pensieve_traces/cooked_test_traces/'
#COOKED_TRACE_FOLDER = './datasets/network_trace/pitree_traces/fcc/'
#COOKED_TRACE_FOLDER = './datasets/network_trace/swift_traces/test/'

SUMMARY_DIR = './results'
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RAND_RANGE = 1000000
LOG_FILE = './results/'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
BITRATE_LEVELS = 6
TOTAL_VIDEO_CHUNK = 49
TOTAL_VIDEO_CHUNKS = 49
CHUNK_TIL_VIDEO_END_CAP = 49.0
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
VIDEO_SIZE_FILE = './datasets/video_trace/video_size_'

RESEVOIR = 5  # BB
CUSHION = 10  # BB
TARGET_BUFFER = [0.5 , 1.0]
DEFAULT_QUALITY = 1  # default video quality without agent

S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
MPC_FUTURE_CHUNK_COUNT = 5
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
