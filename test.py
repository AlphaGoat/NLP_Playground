import argparse
import os


import tensorflow as tf





def main(flags):
    '''Adopted from Ian W. McQuaid'''
    print("Script starting...")

    # Set the GPUs we want the script to use/see
    print("GPU List = " + str(flags.gpu_list))
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_list

    with tf.device('/cpu:0'):
        if flags.use_test_problem:
            # TODO: establish general datapath for test data and real data
            pass
        else:
            data = DatasetGenerator_PtToEng
            print("Building generator...")
            train_generator = iter(data.train_dataset)
            val_generator = iter(data.val_dataset)
            print("Generator built.")

        if flags.run_pipeline_test
            with tf.Session() as sess:
                pt_batch, en_batch = next(train_generator)
                print("pt_batch shape = " + str(pt_batch.shape))
                print("en_batch_shape = " + str(en_batch.shape))


        # Instantiate positional encoder

        # Construct generators yielding data batches

if __name__ == "__main__":

    # Instantiating arg parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--vocab_size', type=int,
                        default=8500,
                        help="Number of unique words in corpus")

    parser.add_argument('--run_name', type=str,
                        default='TNN_tensorboard_valid_test',
                        help="The name of this run, to be used on the weights file and Tensorboard logs")

    # Transformer neural network specific hyperparameters
    parser.add_argument('--dim_model', type=int,
                        default=512,
                        help="Dimension of embeddings")

    parser.add_argument('--num_layers', type=int,
                        default=6,
                        help="Number of transformer layers to instantiate model with")

    parser.add_argument('--dff', type=int,
                        default=2048,
                        help="Dimensionality of the inner layer")

    parser.add_argument('--num_heads', type=int,
                        default=8,
                        help="Number of parallel attention layers to use")

    parser.add_argument('--learning_rate', type=float,
                        default=1e-4,
                        help='Initial learning rate')

    parser.add_argument('--num_training_epochs', type=int,
                         default=10000,
                         help='Number of epochs to train model')

    parser.add_argument('--patience', type=int,
                        default=100,
                        help='Number of epochs without improvement to trigger early stopping')

    parser.add_argument('--dataset_buffer_size', type=int,
                        default=124,
                        help='Number of images to prefetch in the input pipeline')

    parser.add_argument('--num_dataset_threads', type=int,
                        default=1,
                        help='Number of threads to be used by the input pipeline')

    parser.add_argument('--batch_size', type=int,
                        )

    # Parse known arguments
    parsed_flags, _ = parser.parse_known_args()

    # call main
    main(parsed_flags)