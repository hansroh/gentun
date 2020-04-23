#!/usr/bin/env python
"""
Create a client which loads MNIST data and waits for jobs
to evaluate models. The rabbitmq service should be running
in 'localhost'.
"""

import os
import sys
import argparse

from gentun import GentunClient
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

parser = argparse.ArgumentParser("Neural Architecture Search Client")
parser.add_argument('-g', '--gpu', type=str, default="0", help="Select GPU Device")
args = parser.parse_args()

if __name__ == '__main__':
    gc = GentunClient(args.gpu, host='223.195.37.85', user='test', password='test')
    gc.work()
