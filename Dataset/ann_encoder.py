import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

import pdb



#class PositionalEncoder(object):
#    '''Positional encoder to give model information about relative
#       position of words in an input sentence. For full details about
#       theory and implementation, see :
#
#       https://github.com/tensorflow/examples/blob/master/community/en/position_encoding.ipynb
#
#    '''
#
#    def __init__(self):


def get_angles(position, i, d_model):
   angle_rates = 1 / np.power(10000, (2 * i //2)) / np.float32(d_model)
   return position * angle_rates

def positional_encoding(position, d_model):
   angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                           np.arange(d_model)[np.newaxis, :],
                           d_model)

   # apply sin to even indices in the array; 2i
   sines = np.sin(angle_rads[:, 0::2])

   # apply cos to odd indices in the array; 2i+1
   cosines = np.cos(angle_rads[:, 1::2])

   pos_encoding = np.concatenate([sines, cosines], axis=-1)

   pos_encoding = pos_encoding[np.newaxis, ...]

   return tf.cast(pos_encoding, dtype=tf.float32)


if __name__ == '__main__':

    pos_encoding = positional_encoding(50, 512)
    print(pos_encoding.shape)

    #pdb.set_trace()
    plt.pcolormesh(pos_encoding[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()