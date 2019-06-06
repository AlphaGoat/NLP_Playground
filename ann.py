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

       author: 1st Lt Peter Thomas
    '''
    def __init__(self, input_size, label_size, learning_rate, d_model, num_heads):


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

    def attention(self, Q, K, V):
        '''Implements attention layer

            :param Q: matrix of set of queries
            :param K: matrix of set of keys
            :param V: matrix of set of values
        '''
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        return tf.nn.softmax(tf.matmul(tf.matmul(Q, K)/tf.math.sqrt(dk)), V)

    def multihead_attention(self, Q, K, V):

        wq = self.weight_variable(self.d_model)
        wk = self.weight_variable(self.d_model)
        wv = self.weight_variable(self.d_model)

        batch_size = tf.shape(Q)[0]


