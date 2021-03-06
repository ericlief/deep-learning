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
                    # args:           add_conv_layer(input_data, n_input_channels, n_filters, filter_shape, stride_shape, pad, name):
                    # out_layer = self.add_conv_layer(self.images, 1, n_filters, filter_shape, stride_shape, pad, 'conv_layer'+str(n_conv_layers))
                    out_layer = tf.layers.conv2d(out_layer, n_filters, filter_size, strides, padding=pad, activation=tf.nn.relu, name='conv_layer_'+str(n_conv_layers))
                    print('cnn', out_layer)
                # M-kernel_size-stride: Add max pooling with specified size and stride. Example: M-3-2
                elif opts[0] == 'M':
                    n_pooling_layers += 1
                    max_pool_sz = int(opts[1])
                    #pool_shape = [1, max_pool_sz, max_pool_sz, 1]
                    strides = int(opts[2])
                    #stride_shape = [1, strides, strides, 1]  #[1,x,y,1]
                    pad = pad if pad else 'SAME'
                    
                    # add layer
                    # out_layer = tf.nn.max_pool(out_layer, pool_shape, stride_shape, pad, name='pooling_layer'+str(n_pooling_layers))
                    out_layer = tf.layers.max_pooling2d(out_layer, max_pool_sz, strides, pad, name='pooling_layer_'+str(n_pooling_layers))
                    print('pool', out_layer)
               
                # F: Flatten inputs
                elif opts[0] == 'F':
                    out_layer = tf.layers.flatten(out_layer, name="flatten")
                    print('flat', out_layer)
                # R: Apply Fully Connected (dense) layer (fc)
                elif opts[0] == 'R':
                    n_hidden_layers += 1
                    fc_size = int(opts[1])
                    out_layer = tf.layers.dense(out_layer, fc_size, activation=tf.nn.relu, name="hidden_layer_"+str(n_hidden_layers))
                    print('fc', out_layer)
                    
                # D: Apply dropout
                elif opts[0] == 'D':
                    n_dropout += 1
                    rate = float(opts[1])
                    out_layer = tf.layers.dropout(out_layer, rate=rate, training=self.is_training, name="dropout"+str(n_dropout))
                    print('drop', out_layer)
                    
            # First model for numeric labels
            logits_labels = tf.layers.dense(out_layer, self.LABELS, activation=None, name="logits")
            print('log lab', logits_labels)
            
            #self.predictions = tf.argmax(logits_labels, axis=1)            
            #print('pred', self.predictions)
            
            # Second model for masks labels: note out_layer should be flattened
            fc_1 = tf.layers.dense(out_layer, 384, activation=tf.nn.relu, name="fcl_mascs_1")
            logits_mask = tf.layers.dense(fc_1, 784, activation=None, name="fcl_mascs_2")
            logits_mask = tf.reshape(logits_mask, [-1, 28, 28, 1])

            self.labels_predictions = tf.argmax(logits_labels, axis=1)
            self.masks_predictions = (tf.sign(logits_mask) + 1) * 0.5

            # Training
            loss1 = tf.losses.sparse_softmax_cross_entropy(self.labels, logits_labels, scope="loss1")
            loss2 = tf.losses.sigmoid_cross_entropy(self.masks, logits_mask, scope="loss2")
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
    parser.add_argument("--cnn", default=None, type=str, help="Description of the CNN architecture.")
		    
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
    with open("{}/fashion_masks_dev.txt".format(args.logdir), "w") as test_file:
        
        while not dev.epoch_finished():
            images, _, _ = dev.next_batch(args.batch_size)
            labels, masks = network.predict(images)
            for i in range(len(labels)):
                print(labels[i], *masks[i].astype(np.uint8).flatten(), file=test_file)
                #with open("fashion_masks_test.txt", "w") as test_file:
    
    with open("{}/fashion_masks_test.txt".format(args.logdir), "w") as test_file:
        
        while not test.epoch_finished():
            images, _, _ = test.next_batch(args.batch_size)
            labels, masks = network.predict(images)
            for i in range(len(labels)):
                print(labels[i], *masks[i].astype(np.uint8).flatten(), file=test_file)


 

 
            ## Construct the network and training operation.
            #n_conv_layers = 0
            #n_pooling_layers = 0
            #n_hidden_layers = 0
            #n_dropout = 0
            ## Process args and get architecture
            #layers = args.cnn.split(",")            
            #out_layer = self.images
            #for layer in layers:
                #opts = layer.split('-')
                
                ## C-filters-kernel_size-stride-padding: Add a convolutional layer with ReLU activation and
                ##   specified number of filters, kernel size, stride and padding. Example: C-10-3-1-same                
                #if opts[0] == 'C':  # e.g. C-10-3-2-same
                    #n_conv_layers += 1
                    #n_filters = int(opts[1]) # e.g. 10
                    #filter_size = int(opts[2]) # e.g. 3 X 3 (sqr)
                    ##filter_shape = [filter_size, filter_size]  # e.g. [3, 3]                 
                    #strides = int(opts[3]) # e.g. # 2 X 2
                    ##stride_shape = [strides, strides]  #[x, y]
                    #pad = opts[4]
                    ## note takes 4d [b, y, x, c]
                    ## add conv layer: 
                    ## args:           add_conv_layer(input_data, n_input_channels, n_filters, filter_shape, stride_shape, pad, name):
                    ## out_layer = self.add_conv_layer(self.images, 1, n_filters, filter_shape, stride_shape, pad, 'conv_layer'+str(n_conv_layers))
                    #out_layer = tf.layers.conv2d(out_layer, n_filters, filter_size, strides, padding=pad, activation=tf.nn.relu, name='conv_layer_'+str(n_conv_layers))
                    
                ## M-kernel_size-stride: Add max pooling with specified size and stride. Example: M-3-2
                #elif opts[0] == 'M':
                    #n_pooling_layers += 1
                    #max_pool_sz = int(opts[1])
                    ##pool_shape = [1, max_pool_sz, max_pool_sz, 1]
                    #strides = int(opts[2])
                    ##stride_shape = [1, strides, strides, 1]  #[1,x,y,1]
                    #pad = pad if pad else 'SAME'
                    
                    ## add layer
                    ## out_layer = tf.nn.max_pool(out_layer, pool_shape, stride_shape, pad, name='pooling_layer'+str(n_pooling_layers))
                    #out_layer = tf.layers.max_pooling2d(out_layer, max_pool_sz, strides, pad, name='pooling_layer_'+str(n_pooling_layers))
               
                ## F: Flatten inputs
                #elif opts[0] == 'F':
                    #out_layer = tf.layers.flatten(out_layer, name="flatten")
               
                ## R: Apply Fully Connected (dense) layer (fc)
                #elif opts[0] == 'R':
                    #n_hidden_layers += 1
                    #fc_size = int(opts[1])
                    #out_layer = tf.layers.dense(out_layer, fc_size, activation=tf.nn.relu, name="hidden_layer_"+str(n_hidden_layers))
               
                ## D: Apply dropout
                #elif opts[0] == 'D':
                    #n_dropout += 1
                    #rate = float(opts[1])
                    #out_layer = tf.layers.dropout(out_layer, rate=rate, training=self.is_training, name="dropout"+str(n_dropout))
                
            #logits = tf.layers.dense(out_layer, self.LABELS, activation=None, name="logits")
            #self.predictions = tf.argmax(logits, axis=1)
        
            ## Training
            #loss = tf.losses.sparse_softmax_cross_entropy(self.labels, logits, scope="loss")
            #global_step = tf.train.create_global_step()
            #self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")            
           
            ##conv_1 = self.images
            ##for i in range(2):
                ##conv_1 = tf.layers.conv2d(conv_1, filters=args.cnn_dim_1, kernel_size=[3,3], strides=2, padding="same", activation=tf.nn.sigmoid, name="conv_1_"+str(i))
                ##print(conv_1)
       
            ##if args.dropout_1:
                ##conv_1 = tf.layers.dropout(conv_1, rate=args.dropout_1, training=self.is_training, name="dropout_1")
            
            ##if args.bn_1:
                ##conv_1 = tf.layers.batch_normalization(conv_1, name='bn_1')
            
            ##if args.max_pool:
                ##conv_1 = tf.layers.max_pooling2d(conv_1, 2, 2, 'valid', name='pool_1') 
            ##if args.ave_pool:
                ##conv_1 = tf.layers.average_pooling2d(conv_1, 2, 2, 'valid', name='pool_1') 
            
            ##for i in range(2):
                ##conv_2 = tf.layers.conv2d(conv_1, filters=args.cnn_dim_2, kernel_size=[3,3], strides=2, padding="same", activation=tf.nn.sigmoid, name="conv_2_"+str(i))
                ##print(conv_2)                     
            
            ##if args.max_pool:
                ##conv_2 = tf.layers.max_pooling2d(conv_2, 2, 2, 'valid', name='pool_2')             
            
            ##if args.pool:
                ##conv_2 = tf.layers.average_pooling2d(conv_2, 2, 2, 'valid', name='pool_2')   
            
            ##if args.dropout_2:
                ##conv_2 = tf.layers.dropout(conv_2, rate=args.dropout_2, training=self.is_training, name="dropout_2")
                
            ##if args.bn_2:
                ##conv_2 = tf.layers.batch_normalization(conv_2, name='bn_2')
               
               
        
            ##flat = tf.layers.flatten(conv_2, name="flatten")
            ##fc_1 = tf.layers.dense(flat, 256, activation=tf.nn.relu, name="fcl_1")
            
            ##if args.dropout_4:
                ##fc_1 = tf.layers.dropout(fc_1, rate=args.dropout_4, training=self.is_training, name="dropout_4")            
            ## Summaries
            #self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            #summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            #self.summaries = {}
            #with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                #self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           #tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
            #with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                #for dataset in ["dev", "test"]:
                    #self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", loss),
                                               #tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]

            ## Initialize variables
            #self.session.run(tf.global_variables_initializer())
            #with summary_writer.as_default():
                #tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    #def train(self, images, labels):
         
         #self.session.run([self.training, self.summaries["train"]], 
                          #feed_dict={self.images: images, self.labels: labels, self.is_training: True})

    #def evaluate(self, dataset, images, labels):
        
        #return self.session.run([self.summaries[dataset], self.accuracy], feed_dict={self.images: images, self.labels: labels, self.is_training: False})

    #def predict(self, images):
        #return self.session.run([self.predictions],
                                #{self.images: images, self.is_training: False})




#if __name__ == "__main__":
    #import argparse
    #import datetime
    #import os
    #import re

    ## Fix random seed
    #np.random.seed(42)

    ## Parse arguments
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    #parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    #parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    #parser.add_argument("--dropout_1", default=0, type=float, help="Dropout rate.")
    #parser.add_argument("--dropout_2", default=0, type=float, help="Dropout rate.")
    #parser.add_argument("--dropout_3", default=0, type=float, help="Dropout rate.")
    #parser.add_argument("--dropout_4", default=0, type=float, help="Dropout rate.")
    #parser.add_argument("--bn_1", default=False, type=bool, help="Batch normalization.")
    #parser.add_argument("--bn_2", default=False, type=bool, help="Batch normalization.")
    #parser.add_argument("--bn_3", default=False, type=bool, help="Batch normalization.")    
    #parser.add_argument("--cnn_dim_1", default=64, type=int, help="RNN cell dimension.")
    #parser.add_argument("--cnn_dim_2", default=128, type=int, help="RNN cell dimension.")
    #parser.add_argument("--cnn_dim_3", default=256, type=int, help="RNN cell dimension.")
    #parser.add_argument("--pool", default=False, type=bool, help="Pooling.")     
    #parser.add_argument("--cnn", default=None, type=str, help="Description of the CNN architecture.")
   
   
   
    #args = parser.parse_args()

    ## Create logdir name
    #args.logdir = "logs/{}-{}-{}".format(
        #os.path.basename(__file__),
        #datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        #",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    #)
    #if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    ## Load the data
    #from tensorflow.examples.tutorials import mnist
    #mnist = mnist.input_data.read_data_sets("mnist-gan", reshape=False, seed=42,
                                            #source_url="https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/mnist-gan/")

    ## Construct the network
    #network = Network(threads=args.threads)
    #network.construct(args)

    ## Train
    #for i in range(args.epochs):
        #while mnist.train.epochs_completed == i:
            #images, labels = mnist.train.next_batch(args.batch_size)
            #network.train(images, labels)

        #_, accuracy = network.evaluate("dev", mnist.validation.images, mnist.validation.labels)
        #print('dev', i, accuracy)     
        ##print(mnist.test.labels)
        ##_, accuracy = network.evaluate("test", mnist.test.images, mnist.test.labels)
        ##print('test', i, accuracy)        

    ## TODO: Compute test_labels, as numbers 0-9, corresponding to mnist.test.images
    #test_labels = network.predict(mnist.test.images)
    #with open("{}/mnist_competition_test.txt".format(args.logdir), "w") as test_file:
        #print('Predicting:')
        #for label in test_labels:
            #print(label)
            #print(label, file=test_file)
