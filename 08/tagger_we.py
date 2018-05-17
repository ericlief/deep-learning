#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import morpho_dataset

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, num_words, num_tags):
        with self.session.graph.as_default():
            if args.recodex:
                tf.get_variable_scope().set_initializer(tf.glorot_uniform_initializer(seed=42))

            # Inputs
            self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens")
            self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")
            self.tags = tf.placeholder(tf.int32, [None, None], name="tags")

            # TODO: Choose RNN cell class according to args.rnn_cell (LSTM and GRU
            # should be supported, using tf.nn.rnn_cell.{BasicLSTM,GRU}Cell). 
            num_units = args.rnn_cell_dim
            if args.rnn_cell == 'RNN':
                cell_fw = tf.nn.rnn_cell.BasicRNNCell(num_units)
                cell_bw = tf.nn.rnn_cell.BasicRNNCell(num_units)
                
            elif args.rnn_cell == 'LSTM':
                cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units)
                cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units)
                
            else:
                cell_fw = tf.nn.rnn_cell.GRUCell(num_units)
                cell_bw = tf.nn.rnn_cell.GRUCell(num_units)
        
            # TODO: Create word embeddings for num_words of dimensionality args.we_dim
            # using `tf.get_variable`.
            #voc_size = len(train.factors[train.FORMS].words)
            word_embeddings = tf.get_variable('word_embeddings', [num_words, args.we_dim])

            # TODO: Embed self.word_ids according to the word embeddings, by utilizing
            # `tf.nn.embedding_lookup`.
            embedded_words = tf.nn.embedding_lookup(word_embeddings, self.word_ids) # which word ids?
            #print('we', embedded_words)
            #print('wids', self.word_ids)
            # TODO: Using tf.nn.bidirectional d_dynamic_rnn, process the embedded inputs.
            # Use given rnn_cell (different for fwd and bwd direction) and self.sentence_lens.
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=embedded_words, sequence_length=self.sentence_lens, dtype=tf.float32)
            #print('out', outputs[0], outputs[1])
            
            # TODO: Concatenate the outputs for fwd and bwd directions (in the third dimension).
            #output_fw, output_bw = outputs
            output = tf.concat(outputs, axis=2)
            #print('concat', output)
            # TODO: Add a dense layer (without activation) with num_tags classes and
            # store result in `output_layer`.
            #num_tags = len(train.factors[train.TAGS].words)
            #print("num tags", num_tags)
            output_layer = tf.layers.dense(output, num_tags) 
            #print('shape output', output_layer)
            # TODO: Generate `self.predictions` using softmax?
            
            self.predictions = tf.argmax(output_layer, axis=2) # 3rd dim!
            #print(self.predictions)
            # TODO: Generate `weights` as a 1./0. mask of valid/invalid words (using `tf.sequence_mask`).
            weights = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)
            #weights = tf.cast(weights, tf.float32)
            #print("weights", weights)
            # Training

            # TODO: Define `loss` using `tf.losses.sparse_softmax_cross_entropy`, but additionally
            # use `weights` parameter to mask-out invalid words.
            loss = tf.losses.sparse_softmax_cross_entropy(labels=self.tags, logits=output_layer, weights=weights)
            #print('loss' , loss)
            self.global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=self.global_step, name="training")

            # Summaries
            self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(self.tags, self.predictions, weights=weights)
            self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=tf.reduce_sum(weights))
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.update_loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.update_accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.current_loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.current_accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train_epoch(self, train, batch_size):
        while not train.epoch_finished():
            # uncomment to stop after 1000 or so steps for higher acc 
            #global_step = tf.train.global_step(self.session, self.global_step)
            #print('global_step', global_step) # prints step
            #if global_step >= 1000:
                #break            
            sentence_lens, word_ids = train.next_batch(batch_size)
            self.session.run(self.reset_metrics)
            #print(word_ids[train.TAGS])
            self.session.run(self.training,
                             {self.sentence_lens: sentence_lens, self.word_ids: word_ids[train.FORMS],
                              self.tags: word_ids[train.TAGS]})
         
            #print(tf.train.get_global_step(self.graph) # gets global_step tensor
            #step = self.session.run(self.global_step)
            #print(step)
            
    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            sentence_lens, word_ids = dataset.next_batch(batch_size)
            #self.session.run([self.update_accuracy, self.update_loss],
                             #{self.sentence_lens: sentence_lens, self.word_ids: word_ids[dataset.FORMS],
                              #self.tags: word_ids[dataset.TAGS]})
            self.session.run([self.update_accuracy, self.update_loss],
                             {self.sentence_lens: sentence_lens, self.word_ids: word_ids[dataset.FORMS],
                              self.tags: word_ids[dataset.TAGS]})                            
        return self.session.run([self.current_accuracy, self.summaries[dataset_name]])[0]
    #{self.sentence_lens: sentence_lens, self.word_ids: word_ids[train.FORMS],
     #self.tags: word_ids[train.TAGS]})

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--recodex", default=False, action="store_true", help="ReCodEx mode.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=64, type=int, help="RNN cell dimension.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train = morpho_dataset.MorphoDataset("czech-cac-train.txt", max_sentences=5000)
    dev = morpho_dataset.MorphoDataset("czech-cac-dev.txt", train=train, shuffle_batches=False)

    #tags = set([])
    #with open("czech-cac-train.txt", 'rt') as f:
        #for line in f:
            #if len(line) > 1: 
                #w,l,t = line.split()
                #tags.add(t)
    #print(len(tags), tags)
    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, len(train.factors[train.FORMS].words), len(train.factors[train.TAGS].words))

    # Train
    for i in range(args.epochs):
        network.train_epoch(train, args.batch_size)

        accuracy = network.evaluate("dev", dev, args.batch_size)
        print("{:.2f}".format(100 * accuracy))
