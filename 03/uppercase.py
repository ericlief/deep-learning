#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

# Loads an uppercase dataset.
# - The dataset either uses a specified alphabet, or constructs an alphabet of
#   specified size consisting of most frequent characters.
# - The batches are generated using a sliding window of given size,
#   i.e., for a character, we generate left `window` characters, the character
#   itself and right `window` characters, 2 * `window` +1 in total.
# - The batches can be either generated using `next_batch`+`epoch_finished`,
#   or all data in the original order can be generated using `all_data`.
class Dataset:
    def __init__(self, filename, window, alphabet):
        self._window = window

        # Load the data
        with open(filename, "r", encoding="utf-8") as file:
            self._text = file.read()
            self.N = len(self._text)
        # Create alphabet_map
        alphabet_map = {"<pad>": 0, "<unk>": 1}
        if not isinstance(alphabet, int):
            for index, letter in enumerate(alphabet):
                alphabet_map[letter] = index
        else:
            # Find most frequent characters
            freqs = {}
            for char in self._text:
                char = char.lower()
                freqs[char] = freqs.get(char, 0) + 1
            # Map char to its freq rank
            most_frequent = sorted(freqs.items(), key=lambda item:item[1], reverse=True)
            for i, (char, freq) in enumerate(most_frequent, len(alphabet_map)):
                alphabet_map[char] = i
                if len(alphabet_map) >= alphabet: break

        # Remap input characters using the alphabet_map
        self._lcletters = np.zeros(len(self._text) + 2 * window, np.uint8)
        self._labels = np.zeros(len(self._text), np.bool)
        for i in range(len(self._text)):
            char = self._text[i].lower()
            if char not in alphabet_map: char = "<unk>"
            self._lcletters[i + window] = alphabet_map[char]
            self._labels[i] = self._text[i].isupper()
            
        
        print("lcletters\n", self._lcletters)
        print("labels ", self._labels)
        print("map\n", alphabet_map)
        
        # Compute alphabet
        self._alphabet = [""] * len(alphabet_map)
        for key, value in alphabet_map.items():
            self._alphabet[value] = key
            
        print("alph\n", self._alphabet)
        print("length of alph ", len(self._alphabet))
              
        self._permutation = np.random.permutation(len(self._text))  # permutation of indices of text
        print("perm\n", self._permutation)
         
    def _create_batch(self, batch_permutation):
        #print("creating batch")
        #print("batch perm\n", batch_permutation)
        batch_windows = np.zeros([len(batch_permutation), 2 * self._window + 1], np.int32)
        for i in range(0, 2 * self._window + 1):
            #print("iter ", i)
            #print(batch_windows[:,i])
            #print("batch perm ", batch_permutation)
            #print("lc - i", self._lcletters[batch_permutation])
            batch_windows[:, i] = self._lcletters[batch_permutation + i]
            #print(batch_windows[:,i])
        return batch_windows, self._labels[batch_permutation]

    #@property
    #def alphabet(self):
        #return self._alphabet 

    @property
    def text(self):
        return self._text

    @property
    def labels(self):
        return self._labels

    #def sizeAlph(self):
        #return len(self.
                   
    def all_data(self):
        return self._create_batch(np.arange(len(self._text)))

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        # [3, 4, 1 , ...], [ 1, 3 ...] are indices
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._create_batch(batch_perm)

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._text))
            return True
        return False


class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.windows = tf.placeholder(tf.int32, [None, 2 * args.window + 1], name="windows")
            self.labels = tf.placeholder(tf.int64, [None], name="labels") # Or you can use tf.int32
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

           
            # Define hidden layers, with dropout if specified
            #hidden_layer = tf.cast(self.windows, tf.float32)
            
            print(self.windows)
            hidden_layer = tf.one_hot(self.windows, depth=args.alphabet_size)
            print("hot ", hidden_layer)
            hidden_layer = tf.contrib.layers.flatten(hidden_layer)
            print("flat ", hidden_layer)
            
            for i in range(0, args.layers):
                if args.activation == "none":
                    hidden_layer = tf.layers.dense(hidden_layer, args.hidden_layer, activation=None, name="hidden_layer"+str(i))
                elif (args.activation == "relu"):
                    hidden_layer = tf.layers.dense(hidden_layer, args.hidden_layer, activation=tf.nn.relu, name="hidden_layer"+str(i))
                elif (args.activation == "tanh"):
                    hidden_layer = tf.layers.dense(hidden_layer, args.hidden_layer, activation=tf.nn.tanh, name="hidden_layer"+str(i))
                elif (args.activation == "sigmoid"):
                    hidden_layer = tf.layers.dense(hidden_layer, args.hidden_layer, activation=tf.nn.sigmoid, name="hidden_layer"+str(i))
                else:
                    print("Error: unknown activation")
                
                print("hidden ", hidden_layer)
                #hidden_layer_dropout = tf.cond(self.is_training, 
                                   #lambda: tf.nn.dropout(hidden_layer, keep_prob=(1 - args.dropout)), # func 1 if true
                                   #lambda: hidden_layer) # func 2 otherwise        

            #output_layer = tf.layers.dense(hidden_layer_dropout, self.labels, activation=None, name="output_layer")
            output_layer = tf.layers.dense(hidden_layer, 2, activation=None, name="output_layer")
            # Training
            print(output_layer)
            
            print(self.labels)
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")
            #loss = tf.losses.sigmoid_cross_entropy(self.labels, output_layer, scope="loss")
            global_step = tf.train.create_global_step()
            
            if args.optimizer == "Adam":
                self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")
            elif args.optimizer == "sgd":
                self.training = tf.train.GradientDescentOptimizer(0.01).minimize(loss, global_step=global_step, name="sgd")
                
            self.predictions = tf.argmax(output_layer, axis=1)
            
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

    def train(self, windows, labels):
        #self.session.run([self.training, self.summaries["train"]], {self.windows: windows, self.labels: labels})
        self.session.run([self.training, self.summaries["train"], ], feed_dict={self.windows: windows, self.labels: labels, self.is_training: True})

    def evaluate(self, dataset, windows, labels):
        #return self.session.run(self.summaries[dataset], {self.windows: windows, self.labels: labels})
        return self.session.run([self.summaries[dataset], self.accuracy], feed_dict={self.windows: windows, self.labels: labels, self.is_training: False})


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(22)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", default="none", type=str, help="Activation function.")
    parser.add_argument("--alphabet_size", default=26, type=int, help="Alphabet size.")
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--window", default=10, type=int, help="Size of the window to use.")
    parser.add_argument("--hidden_layer", default=200, type=int, help="Size of the hidden layer.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
    parser.add_argument("--learning_rate_final", default=None, type=float, help="Final learning rate.")
    parser.add_argument("--momentum", default=None, type=float, help="Momentum.")
    parser.add_argument("--optimizer", default="sgd", type=str, help="Optimizer to use.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--layers", default=1, type=int, help="Number of layers.")
    #parser.add_argument("--training_size", default=5000, type=int, help="Number of training images.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    #train = Dataset("text.txt", args.window, alphabet=args.alphabet_size)
    train = Dataset("uppercase_data_train.txt", args.window, alphabet=args.alphabet_size)
    dev = Dataset("uppercase_data_dev.txt", args.window, alphabet=train._alphabet)
    test = Dataset("uppercase_data_test.txt", args.window, alphabet=train._alphabet)

    #Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        while not train.epoch_finished():
            windows, labels = train.next_batch(args.batch_size)
            #print("win", windows)
            #print("lab", labels)
            network.train(windows, labels)

        dev_windows, dev_labels = dev.all_data()
        network.evaluate("dev", dev_windows, dev_labels)

    # TODO: Generate the uppercased test set
