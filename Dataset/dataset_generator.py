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

        :return: the exoected number of batches that will be produced by this generator
        """
        return self.num_text_bodies // self.batch_size

    def get_dataset(self):
        return self.dataset

    def get_iterator(self):
        # Create and return iterator
        return self.dataset.make_one_shot_iterator()

    def _parse_data(self, example_proto):


    def build_dataset(words, n_words):

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


    def generate_batch(data, batch_size, num_skips, skip_window):
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