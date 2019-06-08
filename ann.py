import tensorflow as tf

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


class Model(object):
    '''Tensorflow implementation of Transformer network described in the
       Google paper "Attention Is All You Need"

       link: https://arxiv.org/pdf/1706.03762.pdf

       code partially adapted from tensorflow tutorial

       link: https://www.tensorflow.org/alpha/tutorials/text/transformer

       author: 1st Lt Peter Thomas
    '''
    def __init__(self, input_size, label_size, learning_rate,
                 d_model, num_heads):

        self.input_size = input_size
        self.label_size = label_size
        self.learning_rate = learning_rate

        self.d_model = d_model
        self.num_heads = num_heads

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

    def pointwise_feed_forward_layer(self, x, W, b):

        first_fc = self.fc(x, 
        second_fc = self.fc(first_fc, tf.get_shape(first_fc[1]), 



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

    def attention(self, Q, K, V, mask=0):
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
        attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(attention_logits, axis=-1)
        output = tf.matmul(attention_weights, V)
        return output, attention_weights

    def multihead_attention(self, Q, K, V):

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

        attention, attention_weights = self.attention(q, k, v)

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
