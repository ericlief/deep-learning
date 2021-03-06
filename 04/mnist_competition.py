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
            # [batch size, w, h, channels=rgb/1 for mono, etc]
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="images")  
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")
            
            # Construct the network and training operation.
            n_conv_layers = 0
            n_pooling_layers = 0
            n_hidden_layers = 0
            n_dropout = 0
            # Process args and get architecture
            layers = args.cnn.split(",")            
            out_layer = self.images
            for layer in layers:
                opts = layer.split('-')
                
                # C-filters-kernel_size-stride-padding: Add a convolutional layer with ReLU activation and
                #   specified number of filters, kernel size, stride and padding. Example: C-10-3-1-same                
                if opts[0] == 'C':  # e.g. C-10-3-2-same
                    n_conv_layers += 1
                    n_filters = int(opts[1]) # e.g. 10
                    filter_size = int(opts[2]) # e.g. 3 X 3 (sqr)
                    #filter_shape = [filter_size, filter_size]  # e.g. [3, 3]                 
                    strides = int(opts[3]) # e.g. # 2 X 2
                    #stride_shape = [strides, strides]  #[x, y]
                    pad = opts[4]
                    # note takes 4d [b, y, x, c]
                    # add conv layer: 
                    
                    out_layer = tf.layers.conv2d(out_layer, n_filters, filter_size, strides, padding=pad, activation=tf.nn.relu, name='conv_layer_'+str(n_conv_layers))
                    
                # M-kernel_size-stride: Add max pooling with specified size and stride. Example: M-3-2
                elif opts[0] == 'M':
                    n_pooling_layers += 1
                    max_pool_sz = int(opts[1])
                    strides = int(opts[2])
                    pad = pad if pad else 'SAME'
                    
                    # add layer
                    out_layer = tf.layers.max_pooling2d(out_layer, max_pool_sz, strides, pad, name='pooling_layer_'+str(n_pooling_layers))
               
                # F: Flatten inputs
                elif opts[0] == 'F':
                    out_layer = tf.layers.flatten(out_layer, name="flatten")
               
                # R: Apply Fully Connected (dense) layer (fc)
                elif opts[0] == 'R':
                    n_hidden_layers += 1
                    fc_size = int(opts[1])
                    out_layer = tf.layers.dense(out_layer, fc_size, activation=tf.nn.relu, name="hidden_layer_"+str(n_hidden_layers))
               
                # D: Apply dropout
                elif opts[0] == 'D':
                    n_dropout += 1
                    rate = float(opts[1])
                    out_layer = tf.layers.dropout(out_layer, rate=rate, training=self.is_training, name="dropout"+str(n_dropout))
                
            logits = tf.layers.dense(out_layer, self.LABELS, activation=None, name="logits")
            self.predictions = tf.argmax(logits, axis=1)
        
            # Training
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, logits, scope="loss")
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

    def train(self, images, labels):
         
         self.session.run([self.training, self.summaries["train"]], 
                          feed_dict={self.images: images, self.labels: labels, self.is_training: True})

    def evaluate(self, dataset, images, labels):
        
        return self.session.run([self.summaries[dataset], self.accuracy], feed_dict={self.images: images, self.labels: labels, self.is_training: False})

    def predict(self, images):
        return self.session.run(self.predictions,
                                {self.images: images, self.is_training: False})




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
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")   
    parser.add_argument("--cnn", default='C-20-5-2-same,D-.5,C-20-3-2-same,F,R-256,D-.5,R-512', type=str, help="Description of the CNN architecture.")
   
   
   
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
    mnist = mnist.input_data.read_data_sets("mnist-gan", reshape=False, seed=42,
                                            source_url="https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/mnist-gan/")

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels)

        _, accuracy = network.evaluate("dev", mnist.validation.images, mnist.validation.labels)
        print('dev', i, accuracy)     
       

    # TODO: Compute test_labels, as numbers 0-9, corresponding to mnist.test.images
    test_labels = network.predict(mnist.test.images)
    with open("{}/mnist_competition_test.txt".format(args.logdir), "w") as test_file:
        print('Predicting:')
        for label in test_labels:
            print(label)
            print(label, file=test_file)
