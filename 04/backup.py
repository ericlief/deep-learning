#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Network:
    
    HEIGHT = 28
    WIDTH = 28
    LABELS = 10

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            #self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="images")
            self.images = tf.placeholder(tf.float32, [None, 784])
            self.images = tf.reshape(self.images, [-1, 28, 28, 1])
            self.labels = tf.placeholder(tf.int64, [None], name="labels")  # why not [None, 10]?
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")
            
            
            # reshape the input data so that it is a 4D tensor.  The first value (-1) tells function to dynamically shape that
            # dimension based on the amount of data passed to it.  The two middle dimensions are set to the image size (i.e. 28
            # x 28).  The final dimension is 1 as there is only a single colour channel i.e. grayscale.  If this was RGB, this
            # dimension would be 3
            
          
            # Computation
            # flattened_images = tf.layers.flatten(self.images, name="flatten")
            
            # TODO: Add layers described in the args.cnn. Layers are separated by a comma and can be:
            # - C-filters-kernel_size-stride-padding: Add a convolutional layer with ReLU activation and
            #   specified number of filters, kernel size, stride and padding. Example: C-10-3-1-same
            # - M-kernel_size-stride: Add max pooling with specified size and stride. Example: M-3-2
            # - F: Flatten inputs
            # - R-hidden_layer_size: Add a dense layer with ReLU activation and specified size. Ex: R-100
            # Store result in `features`.
            
            n_in_channels = 1  # greyscale
            conv_layer = None
            hidden_layer = None  # no need to flatten?
            pool_layer = None
            layers = args.cnn.split(",")
            for layer in layers:
                opts = layer.split('-')
                
                if opts[0] == 'C':  # e.g. C-10-3-2-same
                    n_filters = int(opts[1]) # e.g. 10
                    filter_size = int(opts[2]) # e.g. 3 X 3 (sqr)
                    filter_shape = [filter_size, filter_size]  # e.g. [3, 3]
                    #filter_shape = [filter_size, filter_size, 1, no_filters] # [y, x, in_c, out_c]
                    #filter = tf.get_variable('filter', filter_shape, initializer=tf.truncated_normal_initializer(stddev=.03, dtype=tf.float32), dtype=tf.float32)
                    
                    #filter = tf.Variable(filter_shape, tf.truncated_normal_initializer(stddev=.03, dtype=tf.float32), dtype=tf.float32, name='filter')

                    #filter = tf.Variable(filter_shape, tf.truncated_normal_initializer(stddev=.03), name='filter')
                                        
                    stride = int(opts[3]) # e.g. # 2 X 2
                    stride_shape = [stride, stride]  #[x, y]
                    pad = opts[4].upper()
                    # note takes 4d [b, y, x, c]
                    # add conv layer: 
                    # args:           add_conv_layer(input_data, n_input_channels, n_filters, filter_shape, stride_shape, pad, name):
                    #conv_layer = self.add_conv_layer(self.images, 1, n_filters, filter_shape, stride_shape, pad, 'conv_layer_1')
                    
                    #hidden_layer = tf.nn.conv2d(hidden_layer, filter, stride_shape, padding, name='conv')
                    #hidden_layer = tf.nn.relu(hidden_layer, name='relu') # add relu 
                
                elif opts[0] == 'M':
                    pooling = True
                    max_pool_sz = int(opts[1])
                    pool_shape = [1, max_pool_sz, max_pool_sz, 1]
                    stride = int(opts[2])
                    stride_shape = [1, stride, stride, 1]  #[1,x,y,1]
                    #pad = pad if pad else 'SAME'
                    
                    # add layer
                     
                    #pool_layer = tf.nn.max_pool(conv_layer, pool_shape, stride_shape, pad, name='max_pool')
                
                elif opts[0] == 'F':
                    pool_layer = tf.layers.flatten(pool_layer, name="flatten")
               
                elif opts[0] == 'H':
                    hidden_layer_sz = int(opts[1])
                    
                    
                    
                    
            # add conv layer: 
            # args:           add_conv_layer(input_data, n_input_channels, n_filters, filter_shape, stride_shape, pad, name):
            conv_layer = self.add_conv_layer(self.images, 1, n_filters, filter_shape, stride_shape, pad, 'conv_layer_1')            
            hidden_layer = tf.layers.dense(pool_layer, hidden_layer_sz, activation=tf.nn.relu, name="hidden_layer")
            output_layer = tf.layers.dense(hidden_layer, self.labels, activation=None, name='output_layer')
            self.predictions = tf.argmax(output_layer, axis=1)

            # Training
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def add_conv_layer(self, input_data, n_input_channels, n_filters, filter_shape, stride_shape, pad, name):
        # setup the filter input shape for tf.nn.conv_2d
        conv_filt_shape = [filter_shape[0], filter_shape[1], n_input_channels, n_filters]
        # setup the stride shape for tf.nn.conv_2d
        stride_shape = [1, stride_shape[0], stride_shape[1], 1]
        # initialise weights and bias for the filter
        weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_W')
        #bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')
    
        # setup the convolutional layer operation
        conv_layer = tf.nn.conv2d(input_data, weights, stride_shape, padding=pad)
    
        # add the bias
        #out_layer += bias
    
        # apply a ReLU non-linear activation
        conv_layer = tf.nn.relu(conv_layer, name='relu')

        return conv_layer
    
    def train(self, images, labels):
        self.session.run([self.training, self.summaries["train"]], {self.images: images, self.labels: labels})

    def evaluate(self, dataset, images, labels):
        accuracy, _ = self.session.run([self.accuracy, self.summaries[dataset]], {self.images: images, self.labels: labels})
        return accuracy


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--cnn", default=None, type=str, help="Description of the CNN architecture.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    from tensorflow.examples.tutorials import mnist
    mnist = mnist.input_data.read_data_sets(".", one_hot=True)  # 28X28, unflattened
    #mnist = mnist.input_data.read_data_sets(".", reshape=False, seed=42, one_hot=True)
    #mnist = mnist.input_data.read_data_sets(".", reshape=False, seed=42)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels)

        network.evaluate("dev", mnist.validation.images, mnist.validation.labels)

    accuracy = network.evaluate("test", mnist.test.images, mnist.test.labels)
    print("{:.2f}".format(100 * accuracy))
