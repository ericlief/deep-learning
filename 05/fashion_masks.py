#!/usr/bin/env python3

#Team: Felipe Vianna and Yuu Sakagushi
# Felipe Vianna: 72ef319b-1ef9-11e8-9de3-00505601122b
# Yuu Sakagushi: d9fbf49b-1c71-11e8-9de3-00505601122b


import numpy as np
import tensorflow as tf

class Dataset:
    def __init__(self, filename, shuffle_batches = True):
        data = np.load(filename)
        self._images = data["images"]
        self._labels = data["labels"] if "labels" in data else None
        self._masks = data["masks"] if "masks" in data else None

        self._shuffle_batches = shuffle_batches
        self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else range(len(self._images))

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def masks(self):
        return self._masks

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._images[batch_perm], self._labels[batch_perm] if self._labels is not None else None, self._masks[batch_perm] if self._masks is not None else None

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else range(len(self._images))
            return True
        return False


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
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.masks = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="masks")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # TODO: Computation and training.
            #
            # The code below assumes that:
            # - loss is stored in `loss`
            # - training is stored in `self.training`
            # - label predictions are stored in `self.labels_predictions` of shape [None] and type tf.int64
            # - mask predictions are stored in `self.masks_predictions` of shape [None, 28, 28, 1] and type tf.float32
            #   with values 0 or 1

            # Network
            #both classification and masks
            
            conv_1 = self.images
            for i in range(10):
                conv_1 = tf.layers.conv2d(conv_1, filters=args.cnn_dim_1, kernel_size=[3,3], strides=1, padding="same", activation=tf.nn.sigmoid, name="conv_1_"+str(i))
                print(conv_1)
       
            if args.dropout_1:
                conv_1 = tf.layers.dropout(conv_1, rate=args.dropout_1, training=self.is_training, name="dropout_1")
            if args.bn_1:
                conv_1 = tf.layers.batch_normalization(conv_1, name='bn_1')
            
            if args.pool:
                conv_1 = tf.layers.average_pooling2d(conv_1, [2,2], 2, 'valid', name='pool')   
            
            #for i in range(2):
                #conv_2 = tf.layers.conv2d(conv_1, filters=args.cnn_dim_2, kernel_size=[3,3], strides=2, padding="same", activation=tf.nn.sigmoid, name="conv_2_"+str(i))
                #print(conv_2)            
            
            #if args.pool:
                #conv_2 = tf.layers.average_pooling2d(conv_2, [2,2], 2, 'valid', name='pool')   
          
            ##conv_2 = tf.layers.conv2d(conv_1, args.cnn_dim_2, kernel_size=[3, 3], strides=2, padding="same", activation=tf.nn.relu, name="conv_2")
            ##print(conv_2)
            
            #if args.dropout_2:
                #conv_2 = tf.layers.dropout(conv_2, rate=args.dropout_2, training=self.is_training, name="dropout_2")
            #if args.bn_2:
                #conv_2 = tf.layers.batch_normalization(conv_2, name='bn_2')
               
               
            #conv_3 = tf.layers.conv2d(conv_2, args.cnn_dim_3, kernel_size=[2, 2], strides=2, padding="same", activation=tf.nn.relu, name="conv_3")
            #print(conv_3)
            
            #if args.dropout_3:
                #conv_3 = tf.layers.dropout(conv_3, rate=args.dropout_2, training=self.is_training, name="dropout_3")
            #if args.bn_3:
                #conv_3 = tf.layers.batch_normalization(conv_2, name='bn_3')
                        
            # conv = tf.layers.conv2d(conv2, filters=10, kernel_size=[5, 5], strides=2, padding="same", use_bias=False, activation=None, name="conv")
            # bn = tf.layers.batch_normalization(inputs=conv, axis=-1, training = self.is_training, name="bn")
            # relu = tf.nn.relu(bn)

            # classification
            #pool1 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=1, name="pool1")
            flat = tf.layers.flatten(conv_1, name="flatten")
            print('flat1', flat)
            fcl_1 = tf.layers.dense(flat, 256, activation=tf.nn.relu, name="fcl_1")
            #if args.dropout_4:
                #fc_1 = tf.layers.dropout(fcl_1, rate=args.dropout_4, training=self.is_training, name="dropout_4")
            #fcl_2 = tf.layers.dense(fcl_1, 512, activation=tf.nn.relu, name="fcl_2")

            #output_layer_1 = tf.layers.dense(fcl_2, self.LABELS, activation=None, name="output_layer_1")
            output_layer_1 = tf.layers.dense(fcl_1, self.LABELS, activation=None, name="output_layer_1")

            # Masks output
            flat_2 = tf.layers.flatten(conv_1, name="flatten2")
            #dense3 = tf.layers.dense(flatten2, 900, activation=tf.nn.relu, name="dense3")
            fcl_mascs_1 = tf.layers.dense(flat_2, 384, activation=tf.nn.relu, name="fcl_mascs_1")
            fcl_mascs_2 = tf.layers.dense(fcl_mascs_1, 784, activation=None, name="fcl_mascs_2")
            output_layer_2 = tf.reshape(fcl_mascs_2, [-1, 28, 28, 1])

            self.labels_predictions = tf.argmax(output_layer_1, axis=1)
            #self.masks_predictions = (tf.round(tf.tanh(output_layer2))+1)*0.5
            self.masks_predictions = (tf.sign(output_layer_2) + 1) * 0.5

            # Training
            loss1 = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer_1, scope="loss1")
            loss2 = tf.losses.sigmoid_cross_entropy(self.masks, output_layer_2, scope="loss2")
            loss = loss1 + loss2
            global_step = tf.train.create_global_step()
            self.training1 = tf.train.AdamOptimizer().minimize(loss1, global_step=global_step, name="training1")
            self.training2 = tf.train.AdamOptimizer().minimize(loss2, global_step=global_step, name="training2")


            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.labels_predictions), tf.float32))
            only_correct_masks = tf.where(tf.equal(self.labels, self.labels_predictions),
                                          self.masks_predictions, tf.zeros_like(self.masks_predictions))
            intersection = tf.reduce_sum(only_correct_masks * self.masks, axis=[1,2,3])
            self.iou = tf.reduce_mean(
                intersection / (tf.reduce_sum(only_correct_masks, axis=[1,2,3]) + tf.reduce_sum(self.masks, axis=[1,2,3]) - intersection)
            )

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy),
                                           tf.contrib.summary.scalar("train/iou", self.iou),
                                           tf.contrib.summary.image("train/images", self.images),
                                           tf.contrib.summary.image("train/masks", self.masks_predictions)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset+"/loss", loss),
                                               tf.contrib.summary.scalar(dataset+"/accuracy", self.accuracy),
                                               tf.contrib.summary.scalar(dataset+"/iou", self.iou),
                                               tf.contrib.summary.image(dataset+"/images", self.images),
                                               tf.contrib.summary.image(dataset+"/masks", self.masks_predictions)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, images, labels, masks):
        self.session.run([self.training1, self.summaries["train"]],
                         {self.images: images, self.labels: labels, self.masks: masks, self.is_training: True})
        self.session.run([self.training2, self.summaries["train"]],
                         {self.images: images, self.labels: labels, self.masks: masks, self.is_training: True})

    def evaluate(self, dataset, images, labels, masks):
        return self.session.run([self.summaries[dataset], self.accuracy, self.iou],
                         {self.images: images, self.labels: labels, self.masks: masks, self.is_training: False})

    def predict(self, images):
        return self.session.run([self.labels_predictions, self.masks_predictions],
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
    parser.add_argument("--dropout_1", default=0, type=float, help="Dropout rate.")
    parser.add_argument("--dropout_2", default=0, type=float, help="Dropout rate.")
    parser.add_argument("--dropout_3", default=0, type=float, help="Dropout rate.")
    parser.add_argument("--dropout_4", default=0, type=float, help="Dropout rate.")
    	
    parser.add_argument("--bn_1", default=False, type=bool, help="Batch normalization.")
    parser.add_argument("--bn_2", default=False, type=bool, help="Batch normalization.")
    parser.add_argument("--bn_3", default=False, type=bool, help="Batch normalization.")    
    parser.add_argument("--cnn_dim_1", default=64, type=int, help="RNN cell dimension.")
    parser.add_argument("--cnn_dim_2", default=128, type=int, help="RNN cell dimension.")
    parser.add_argument("--cnn_dim_3", default=256, type=int, help="RNN cell dimension.")
    parser.add_argument("--pool", default=False, type=bool, help="Pooling.")    
		    
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
                  for key, value in sorted(vars(args).items()))).replace("/", "-")
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train = Dataset("fashion-masks-train.npz")
    print(len(train.masks[0]))
    dev = Dataset("fashion-masks-dev.npz")
    test = Dataset("fashion-masks-test.npz", shuffle_batches=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    best_acc = 0
    best_mask = 0
    for i in range(args.epochs):
        while not train.epoch_finished():
            images, labels, masks = train.next_batch(args.batch_size)
            network.train(images, labels, masks)

        result = network.evaluate("dev", dev.images, dev.labels, dev.masks)
        
        if i%5==0:
            print("----------------------", i, "/", args.epochs, "epochs")

        if result[1] >= best_acc:
            best_acc = result[1]
            print("Accuracy: {:.2f}".format(100 * result[1]), 'Best!!!')
        else:
            print("Accuracy: {:.2f}".format(100 * result[1]))

        if result[2] >= best_mask:
            best_mask = result[2]
            print("Mask iou: {:.2f}".format(100 * result[2]), 'Best!!!')
        else:
            print("Mask iou: {:.2f}".format(100 * result[2]))
        print("---------------")

    # Predict test data
    #with open("fashion_masks_test.txt", "w") as test_file:
    with open("{}/fashion_masks_test.txt".format(args.logdir), "w") as test_file:
        
        while not test.epoch_finished():
            images, _, _ = test.next_batch(args.batch_size)
            labels, masks = network.predict(images)
            for i in range(len(labels)):
                print(labels[i], *masks[i].astype(np.uint8).flatten(), file=test_file)
