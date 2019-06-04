"""
Dataset generator, to be used with tensorflow Dataset API in form of TFRecords file

Author: 1st Lt Peter Thomas
Date: 29 May 2019
"""

import tensorflow as tf

import collections
import random
import numpy as np

# Code sourced from https://adventuresinmachinelearning.com/word2vec-tutorial-tensorflow/

class text_DatasetGenerator(object):

    def __init__(self,
                 tfrecord_name,
                 num_text_bodies,
                 augment=False,
                 shuffle=False,
                 batch_size=4,
                 num_threads=1,
                 buffer=30,
                 encoding_function=None,
                 return_filename=False,
                 cache_dataset_memory=False,
                 cache_dataset_file=False,
                 cache_name="",
                 ):

        self.num_text_bodies = num_text_bodies

    def __len__(self):
        """
        The "length of the generator is the number of batches expected.

        :return: the expected number of batches that will be produced by this generator
        """
        return self.num_text_bodies // self.batch_size

    def get_dataset(self):
        return self.dataset

    def get_iterator(self):
        # Create and return iterator
        return self.dataset.make_one_shot_iterator()

    def _parse_data(self, example_proto):


    def build_dataset(self, example_proto, n_words):
        """
        First step of the generator/augmentation chain.
        :param example_proto: Example from a TFRecord file
        :param n_words: number of
        :return:
        """
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(n_words - 1))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)

        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
                unk_count += 1
            data.append(index)

        count[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(),
                                    dictionary.keys()))

        return data, count, dictionary, reversed_dictionary


    def generate_batch(self, data, batch_size, num_skips, skip_window):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        context = np.ndarry(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1
        buffer = collections.deque(maxlen=span)

        for _ in range(span):
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)

        for i in range(batch_size // num_skips):
            target = skip_window
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                context[i * num_skips + j, 0] = buffer[target]

            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)


        def embedding(self, valid_size, valid_window, batch_size, embedding_size,
                      skip_window, num_skips):
            """
            Creates word embeddings for given text.
            :param valid_size: random set of words to evaluate similarity on
            :param batch_size: only pick dev samples in the head of the distribution
            :param embedding_size: dimensions of the embedding vector
            :param skip_window: How many words to consider left and right
            :param num_skips: How many times to reuse an input to generate a context
            :return:
            """
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, )

            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
            )

            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Constructing variables for softmax
            weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                            stddev=1.0 / np.sqrt(embedding_size)))
            biases = tf.Variable(tf.zeros([vocabulary_size]))
            hidden_out = tf.matmul(embed, tf.transpose(weights) + biases)

            # convert train_context to a one-hot format
            train_one_hot = tf.one_hot(train_context, vocabulary_size)
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out,
                                                labels=train_one_hot))
            # Construct the SGD optimizer using a learning rate of 1.0
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(cross_entropy)

            # Compute the cosine similarity between minibatch examples and all embeddings
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm

            valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, valid_dataset
            )

            similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True
            )



