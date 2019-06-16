import tensorflow as tf

import tensorflow_datasets as tfds

# Dataset generator as described on Tensorflow website
# link: https://www.tensorflow.org/beta/tutorials/text/transformer

class DatasetGenerator_PtToEng(object):

    def __init__(self, target_vocab_size=2**13, max_length=40,
                buffer_size=20000, batch_size=64):

        examples, self.metadata = tfds.load(
                'ted_hrlr_translate/pt_to_en', 
                with_info=True, as_supervised=True)
        self.train_examples, self.val_examples = (examples['train'], 
                                examples['validation'])
        self.train_examples = self.train_examples.make_one_shot_iterator()
        self.val_examples = self.val_examples.make_one_shot_iterator
        self.next_train_element = self.train.examples.get_next()
        self.target_vocab_size = target_vocab_size
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # Generate word tokenizers for training dataset
        #self.tokenizer_en, self.tokenizer_pt = self.tokenize_training_set(
        #        self.train_examples, self.target_vocab_size)
        self.tokenizer_en = None
        self.tokenizer_pt = None

        # Generate training dataset
        self.train_dataset = self.train_examples.map(self.tf_encode)
        self.train_dataset = self.train_dataset.filter(
                self.filter_max_length(max_length=self.max_length))

        self.train_dataset = self.train_dataset.cache()
        self.train_dataset = self.train_dataset.shuffle(
                self.buffer_size).padded_batch(self.batch_size,
                padded_shapes=([-1], [-1]))
        self.train_dataset = self.train_dataset.prefetch(
                tf.data.experimental.AUTOTUNE)

        # Generate validation dataset
        self.val_dataset = self.val_examples.map(self.tf_encode)
        self.val_dataset = self.val_dataset.filter(
                self.filter_max_length).padded_batch(
                self.batch_size, padded_shapes=([-1], [-1]))


    def tokenize_dataset(self, train_examples,
            target_vocab_size):
        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                (en.numpy() for pt, en in train_examples),
                target_vocab_size=self.target_vocab_size)
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                (pt.numpy() for pt, en in train_examples),
                target_vocab_size=self.target_vocab_size)

        return tokenizer_en, tokenizer_pt

    # Add start and end token to input and target
    def encode(self, lang1, lang2, tokenizer_pt, tokenizer_en):
        '''Add start and end token to input and target'''
        lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
                    lang1.numpy()) + [tokenizer_pt.vocab_size+1]
        lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
                    lang2.numpy()) + [tokenizer_en.vocab_size+1]

        return lang1, lang2

    def filter_max_length(self, x, y, max_length=40):
        '''Drop examples with a length over max_length number of
           tokens
        '''
        return tf.logical_and(tf.size(x) <= max_length,
                tf.size(y) <= max_length)

    def tf_encode(self, pt, en):
        '''Encoding function run inside tf.py_function that 
           recieves an eager tensor with numpy attribute containing 
           string value.
        '''
        return tf.py_function(encode, [pt, en], [tf.int64, tf.int64])

#def tokenize_training_set(train_examples,
#        target_vocab_size):
#    tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
#        (en.numpy() for pt, en in train_examples),
#        target_vocab_size=target_vocab_size)
#
#    tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
#        (pt.numpy() for pt, en in train_examples),
#        target_vocab_size=target_vocab_size)
#
#    return tokenizer_en, tokenizer_pt

if __name__ == '__main__':

    dataset = DatasetGenerator_PtToEng()
    list_of_training_elements = list()
    with tf.Session as sess:
        while True:
            try:
                train_element = sess.run(dataset.next_train_element)
                list_of_training_elements.append(train_element)
            except tf.errors.OutOfRangeError:
                break
        # Generate tokenizers
        tokenizer_en, tokenizer_pt = sess.run(
                dataset.tokenize_dataset(list_of_training_elements,
                dataset.target_vocab_size))

        dataset.tokenizer_en = tokenizer_en
        dataset.tokenizer_pt = tokenizer_pt

        sample_string = 'Transformer is awesome.'

        tokenized_string = tokenizer_en.encode(sample_string)
        print('The original string: {}'.format(tokenized_string))

        original_string = tokenizer_en.decode(tokenizer_string)
        print('The original string: {}'.format(original_string))

        assert original_string == sample_string

        # words can be broken into subwords if the word is not included
        # in the dictionary
        for ts in tokenized_string:
            print('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))

        pt_batch, en_batch = next(iter(val_dataset))
        print(pt_batch, en_batch)





    #examples, metadata = tfds.load(
    #    'ted_hrlr_translate/pt_to_en',
    #    with_info=True, as_supervised=True)
    #train_examples, val_examples = (examples['train'],
    #                                          examples['validation'])
    #tokenizer_en, tokenizer_pt = tokenize_training_set(train_examples, 2**13)
    #sample_string = 'Transformer is awesome.'
    #tokenized_string = tokenizer_en.encode(sample_string)
    #print('Tokenized string is {}.'.format(tokenized_string))

    #original_string = tokenizer_en.decode(tokenized_string)
    #print('The original string: {}'.format(original_string))


