import argparse

def parse_arg():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-abr', type=str, default='bb', help='abr')
    
    args = parser.parse_args()
    
    return args