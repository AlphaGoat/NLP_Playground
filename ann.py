import tensorflow as tf
import numpy as np

import argparse

import os
import functools

# Custom modules
#from Dataset.dataset_generator_porteng_translate import DatasetGenerator_PtToEng
#from Dataset.tnn_encoder import PositionalEncoder

slim = tf.contrib.slim

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    Decorator source:
    https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator



@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    Decorator source:
    https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


def print_tensor_shape(tensor, string):
    '''
    input: tensor and string to describe it
    
    borrowed from justin r. fletcher
    '''

    if __debug__:
        print('DEBUG' + string, tensor.get_shape())


class Model(object):
    '''Tensorflow implementation of Transformer network described in the
       Google paper "Attention Is All You Need"

       link: https://arxiv.org/pdf/1706.03762.pdf

       code partially adapted from tensorflow tutorial

       link: https://www.tensorflow.org/alpha/tutorials/text/transformer

       author: 1st Lt Peter Thomas
    '''
    def __init__(self, input_size, label_size, learning_rate,
                 d_model, num_heads, enqueue_threads, val_enqueue_threads,
                 data_dir, train_file, validation_file):

        self.input_size = input_size
        self.label_size = label_size
        self.learning_rate = learning_rate

        self.d_model = d_model
        self.num_heads = num_heads

        self.enqueue_threads = enqueue_threads
        self.val_enqueue_threads = val_enqueue_threads
        self.data_dir = data_dir
        self.train_file = train_file
        self.validation_file = validation_file

        assert d_model % num_heads == 0

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor
        (for TensorBoard visualization)."""

        with tf.name_scope('summaries'):

            mean = tf.reduce_mean(var)

            tf.summary.scalar('mean', mean)

            with tf.name_scope('stddev'):

                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

            tf.summary.scalar('stddev', stddev)

            tf.summary.scalar('max', tf.reduce_max(var))

            tf.summary.scalar('min', tf.reduce_min(var))

            tf.summary.histogram('histogram', var)

        return

    def print_out(self, q, k, v):
        '''Print the attention weights and the output'''
        temp_out, temp_attn = attention(
                        q, k, v, None)
        print('Attention weights are:')
        print(temp_attn)
        print('Output is:')
        print(temp_out)

    def create_padding_mask(self, seq):
        '''Mask all pad tokens in batch of sequence. Ensures model
           doesn't treat padding as input
        '''
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions so that we can add the padding
        # to the attention logits
        return seq[:, tf.newaxis, tf.newaxis, :] # (batch_size, 1, 1, seq_len)

    def create_look_ahead_mask(self, size):
        '''Masks future tokens in sequence. Indicates which entries
           should not be used
        '''
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask # (seq_len, seq_len)

    def weight_variable(self, shape):

        initial = tf.truncated_normal(shape, stddev=0.1)
        self.variable_summaries(initial)
        return tf.Variable(initial)

    def bias_variable(self, shape):

        initial = tf.constant(0.1, shape=shape)
        self.variable_summaries(initial)
        return tf.Variable(initial)

    def feed_forward_layer(self, x, W, b):
        return tf.nn.relu(tf.matmul(x, W) + b)

    def pointwise_feed_forward_layer(self, x, W1, b1, W2, b2):

        first_fc = self.fc(x, W1, b1)
        second_fc = self.fc(first_fc, W2, b2, relu=False)
        return second_fc

    def fc(self, x, num_in, num_out, name, relu=True):
        '''Create a fully connected layer.
           Adopted from Justin R. Fletcher'''
        with tf.variable_scope(name) as scope:
            # Create tf variables for the weights and biases
            weights = tf.get_variable('weights', shape=[num_in, num_out],
                            trainable=True)
            biases = tf.get_variable('biases', [num_out], trainable=True)

            # Matrix multiply weights and inputs and add bias
            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu
        else:
            return act

    def attention(self, Q, K, V, mask=None):
        '''Implements attention layerA
            Note: Q, K, and V must have matching leading dimensions
                  K and V must have matching penultimate dimensions
                  (i.e., seq_len_k = seq_len_v)

            :param Q: matrix of set of queries
            :param K: matrix of set of keys
            :param V: matrix of set of values
        '''
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        attention_logits = tf.matmul(Q, K,
                    transpose_b=True) / tf.math.sqrt(dk)
        if mask: attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(attention_logits, axis=-1)
        output = tf.matmul(attention_weights, V)
        return output, attention_weights

    def multihead_attention(self, Q, K, V, mask):

        wq = self.weight_variable(tf.shape(Q)[1], self.d_model)
        wk = self.weight_variable(tf.shape(K)[1], self.d_model)
        wv = self.weight_variable(tf.shape(V)[1], self.d_model)

        bq = self.bias_variable(self.d_model)
        bk = self.bias_variable(self.d_model)
        bv = self.bias_variable(self.d_model)

        aq = self.feed_forward_layer(Q, wq, bq)
        ak = self.feed_forward_layer(K, wk, bk)
        av = self.feed_forward_layer(V, wv, bv)

        batch_size = tf.shape(Q)[0]

        q = self.split_heads(aq, batch_size)
        k = self.split_heads(ak, batch_size)
        v = self.split_heads(av, batch_size)

        attention, attention_weights = self.attention(q, k, v, mask)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(attention,
                        (batch_size, -1, self.d_model))

        wa = self.weight_variable(tf.shape(concat_attention)[1], self.d_model)
        ba = self.weight_variable(tf.shape(self.d_model))

        output = self.feed_forward_layer(concat_attention, wa, ba)

        return output, attention_weights

    def split_heads(self, x, batch_size):

        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def encoder_layer(self, x, rate):
        '''An encoder layer consists of the following sublayers:
                1. Multi-head attention (with padding''''' \

        # TODO: complete pure TensorFlow implementation of encoder layer
        attn_output, _ = self.multihead_attention(x, x, x, mask)
        attn_output = tf.nn.Dropout(attn_output, rate)
        output1 = tf.layers.batch_normalization(attn_output, epsilon=1e-6)

        W_ffn = self.weight_variable()
        b_ffn = self.bias_variable()
        ffn_output = self.pointwise_feed_forward_layer(output1, )


    def decoder_layer(self, x, rate, look_ahead_mask, padding_mask):

        # TODO: complete pure TensorFlow implementation of decoder layer
        attn1, attn_W = self.multihead_attention(x, x, x, look_ahead_mask)

    def encoder(self, num_layers, input=None, input_vocab_size=None, rate=0.1):
        '''
        input: tensor of input corpus. if none, uses instantiation input
        output: tensor of computed logits
        '''
        # TODO: complete pure tensorflow implementation of encoder, based on Justin's code
        ###############################

        print_tensor_shape(self.stimulus_placeholder, 'corpus shape')
        print_tensor_shape(self.target_placeholder, 'label shape')

        # resize the image tensors to add channels, 1 in this case
        # required to pass the images to various layers upcoming in the graph
        images_re = tf.reshape(self.stimulus_placeholder, [-1, 28, 28, 1])
        print_tensor_shape(images_re, 'reshaped images shape')


        pos_encoding = positional_encoding(input_vocab_size, self.d_model)

        for n in range(num_layers):
            with tf.name_scope('encoding_layer' + n):
                h_enc_output = self.encoder_layer(h_enc_output, rate)

        # Convolution layer.
        with tf.name_scope('Conv1'):

            # weight variable 4d tensor, first two dims are patch (kernel) size
            # 3rd dim is number of input channels, 4th dim is output channels
            W_conv1 = self.weight_variable([5, 5, 1, 32])
            b_conv1 = self.bias_variable([32])
            h_conv1 = tf.nn.relu(self.conv2d(images_re, W_conv1) + b_conv1)
            print_tensor_shape(h_conv1, 'Conv1 shape')

        # Pooling layer.
        with tf.name_scope('Pool1'):

            h_pool1 = self.max_pool_2x2(h_conv1)
            print_tensor_shape(h_pool1, 'MaxPool1 shape')

        # Conv layer.
        with tf.name_scope('Conv2'):

            W_conv2 = self.weight_variable([5, 5, 32, 64])
            b_conv2 = self.bias_variable([64])
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
            print_tensor_shape(h_conv2, 'Conv2 shape')

        # Pooling layer.
        with tf.name_scope('Pool2'):

            h_pool2 = self.max_pool_2x2(h_conv2)
            print_tensor_shape(h_pool2, 'MaxPool2 shape')

        # Fully-connected layer.
        with tf.name_scope('fully_connected1'):

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            print_tensor_shape(h_pool2_flat, 'MaxPool2_flat shape')

            W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
            b_fc1 = self.bias_variable([1024])

            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            print_tensor_shape(h_fc1, 'FullyConnected1 shape')

        # Dropout layer.
        with tf.name_scope('dropout'):

            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Output layer (will be transformed via stable softmax)
        with tf.name_scope('readout'):

            W_fc2 = self.weight_variable([1024, 10])
            b_fc2 = self.bias_variable([10])

            readout = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            print_tensor_shape(readout, 'readout shape')

        return readout
        ###############################

    def decoder(self):
        #TODO: complete pure tensorflow implementation of transformer
        #      decoder based on Justin's code
        pass


############################################################
# DEBUGGING FUNCTIONS, DELETE WHEN FINISHED
############################################################
def multihead_attention(Q, K, V, d_model, mask=0):

    print("Debug code:")
    tf.print(tf.shape(Q)[1])

    wq = weight_variable(tf.shape(Q)[1], d_model)
    wk = weight_variable(tf.shape(K)[1], d_model)
    wv = weight_variable(tf.shape(V)[1], d_model)

    bq = bias_variable(d_model)
    bk = bias_variable(d_model)
    bv = bias_variable(d_model)

    aq = feed_forward_layer(Q, wq, bq)
    ak = feed_forward_layer(K, wk, bk)
    av = feed_forward_layer(V, wv, bv)

    batch_size = tf.shape(Q)[0]

    q = split_heads(aq, batch_size)
    k = split_heads(ak, batch_size)
    v = split_heads(av, batch_size)

    attention, attention_weights = attention(q, k, v, mask)

    attention = tf.transpose(attention, perm=[0, 2, 1, 3])

    concat_attention = tf.reshape(attention,
                    (batch_size, -1, d_model))

    wa = weight_variable(tf.shape(concat_attention)[1], d_model)
    ba = weight_variable(tf.shape(d_model))

    output = feed_forward_layer(concat_attention, wa, ba)

    return output, attention_weights

def attention(Q, K, V, mask=0):
    '''Implements attention layerA
        Note: Q, K, and V must have matching leading dimensions
              K and V must have matching p:w
              enultimate dimensions
              (i.e., seq_len_k = seq_len_v)

        :param Q: matrix of set of queries
        :param K: matrix of set of keys
        :param V: matrix of set of values
    '''
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    dk = tf.Print(dk, [dk], "printing out dk: ")
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    matmul_qk = tf.Print(matmul_qk, [matmul_qk, tf.shape(matmul_qk)], "printing out matmul_qk: ")

    attention_logits = matmul_qk / tf.math.sqrt(dk)

    attention_logits = tf.Print(attention_logits, [attention_logits, tf.shape(attention_logits)], "printing out attention_logits: ")

    if mask: attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(attention_logits, axis=-1)
    attention_weights = tf.Print(attention_weights, [attention_weights], "printing out attention_weights: ")
    output = tf.matmul(attention_weights, V)
    return output, attention_weights



def weight_variable(shape):

    initial = tf.truncated_normal(shape, stddev=0.1)
    self.variable_summaries(initial)
    return tf.Variable(initial)


def bias_variable(shape):

    initial = tf.constant(0.1, shape=shape)
    self.variable_summaries(initial)
    return tf.Variable(initial)

def create_padding_mask(seq):
    '''Mask all pad tokens in batch of sequence. Ensures model
       doesn't treat padding as input
    '''
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions so that we can add the padding
    # to the attention logits
    return seq[:, tf.newaxis, tf.newaxis, :] # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    '''Masks future tokens in sequence. Indicates which entries
       should not be used
    '''
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask # (seq_len, seq_len)

def print_out(q, k, v):
    '''Print the attention weights and the output'''
    temp_out, temp_attn = attention(
                    q, k, v, None)
    print('Attention weights are:')
    a = tf.print(temp_attn)
    print('Output is:')
    b = tf.print(temp_out)

if __name__ == '__main__':

        ########## Debug attention ###############
    np.set_printoptions(suppress=True)

    temp_k = tf.constant([[10,0,0],
                          [0,10,0],
                          [0,0,10],
                          [0,0,10]], dtype=tf.float32)

    temp_v = tf.constant([[   1,0],
                          [  10,0],
                          [ 100,5],
                          [1000,6]], dtype=tf.float32)

    # This query aligns with the second key
    # so the second value will be returned
    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)
    with tf.Session() as sess:
    #    a = tf.print(output, [output], "#This is the attention output")
    #    b = tf.print(a, [weights, output], "#These are the attention weights")
    #    with tf.control_dependencies([a,b]):
    #        test_out = tf.debugging.assert_type(output, tf.float32)
            output, weights = sess.run(attention(temp_q, temp_k, temp_v))


        ######### DEBUG look ahead mask ############3
    #x = tf.random.uniform((1, 3))
    #temp = create_look_ahead_mask(x.shape[1])
    #a = tf.print(temp, [temp], "#Debugging")
    
    
        ######### DEBUG padding mask ############3
    #x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    #seq = create_padding_mask(x)
    #a = tf.print(seq, [seq], "#Debugging")


        ####### DEBUG Multihead Attention ##############
    #y = tf.random.uniform((1, 60, 512))
    #with tf.Session() as sess:
    #    sess.run(multihead_attention(y, y, y, 512, mask=0))
    #    print("Printing y:")
    #    tf.print(y)
    #    #test_output, attn = multihead_attention(y, y, y, 512, mask=0)
    #    #test_output.shape, attn.shape

    #with tf.Session() as sess:
    #    sess.run(a)
