import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize

import tensorflow as tf
from tensorboard import summary as summary_lib

import numpy as np
from PIL import Image
import cv2
import io
import time


def print_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return "%d Hours %02d Minutes %02.2f Seconds" % (h, m, s)
    elif m > 0:
        return "%2d Minutes %02.2f Seconds" % (m, s)
    else:
        return "%2.2f Seconds" % s
