#!/usr/bin/env python3

#Team: Felipe Vianna and Yuu Sakagushi
# Felipe Vianna: 72ef319b-1ef9-11e8-9de3-00505601122b
# Yuu Sakagushi: d9fbf49b-1c71-11e8-9de3-00505601122b

import numpy as np
import tensorflow as tf

import nli_dataset

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, num_words, num_chars, num_languages):
        with self.session.graph.as_default():
            # Inputs
            self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens")
            self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")
            self.charseqs = tf.placeholder(tf.int32, [None, None], name="charseqs")
            self.charseq_lens = tf.placeholder(tf.int32, [None], name="charseq_lens")
            self.charseq_ids = tf.placeholder(tf.int32, [None, None], name="charseq_ids")
            self.languages = tf.placeholder(tf.int32, [None], name="languages")

            # TODO: Training.
            # Define:
            # - loss in `loss`
            # - training in `self.training`
            # - predictions in `self.predictions`

            ### Taken from tagger_cnne.py ###

            # TODO(we): Choose RNN cell class according to args.rnn_cell (LSTM and GRU
            # should be supported, using tf.nn.rnn_cell.{BasicLSTM,GRU}Cell).
            rnn_cell_dim = 128
            we_dim = 128
            cle_dim = 64
            cnne_max = 5
            cnne_filters = 16

            # fwd = tf.nn.rnn_cell.BasicLSTMCell(rnn_cell_dim)
            # bwd = tf.nn.rnn_cell.BasicLSTMCell(rnn_cell_dim)
            fwd = tf.nn.rnn_cell.GRUCell(rnn_cell_dim)
            bwd = tf.nn.rnn_cell.GRUCell(rnn_cell_dim)


            # TODO(we): Create word embeddings for num_words of dimensionality args.we_dim
            # using `tf.get_variable`.
            we = tf.get_variable("var", shape=[num_words, we_dim])

            # TODO(we): Embed self.word_ids according to the word embeddings, by utilizing
            # `tf.nn.embedding_lookup`.
            embeded_we = tf.nn.embedding_lookup(params=we, ids=self.word_ids)

            # Convolutional word embeddings (CNNE)

            # TODO: Generate character embeddings for num_chars of dimensionality args.cle_dim.
            cnne = tf.get_variable("var_cle", shape=[num_chars, cle_dim])

            # TODO: Embed self.charseqs (list of unique words in the batch) using the character embeddings.
            embeded_cle = tf.nn.embedding_lookup(params=cnne, ids=self.charseqs)

            # TODO: For kernel sizes of {2..args.cnne_max}, do the following:
            # - use `tf.layers.conv1d` on input embedded characters, with given kernel size
            #   and `args.cnne_filters`; use `VALID` padding, stride 1 and no activation.
            # - perform channel-wise max-pooling over the whole word, generating output
            #   of size `args.cnne_filters` for every word.
            features=[]
            for k in range(2, cnne_max+1):
                conv = tf.layers.conv1d(inputs=embeded_cle, filters=cnne_filters, kernel_size=k,
                                            strides=1, padding='valid', activation=None)

                pool = tf.reduce_max(input_tensor=conv, axis=1)
                features.append(pool)

            # TODO: Concatenate the computed features (in the order of kernel sizes 2..args.cnne_max).
            # Consequently, each word from `self.charseqs` is represented using convolutional embedding
            # (CNNE) of size `(args.cnne_max-1)*args.cnne_filters`.
            concat_cnne = tf.concat(features, axis=1)

            # TODO: Generate CNNEs of all words in the batch by indexing the just computed embeddings
            # by self.charseq_ids (using tf.nn.embedding_lookup).
            embeded_cnne = tf.nn.embedding_lookup(params=concat_cnne, ids=self.charseq_ids)

            # TODO: Concatenate the word embeddings (computed above) and the CNNE (in this order).
            embeded = tf.concat([embeded_we, embeded_cnne], axis=2)

            # TODO(we): Using tf.nn.bidirectional_dynamic_rnn, process the embedded inputs.
            # Use given rnn_cell (different for fwd and bwd direction) and self.sentence_lens.
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwd, cell_bw=bwd, inputs=embeded,
                                                    sequence_length=self.sentence_lens, dtype=tf.float32)

            # TODO(we): Concatenate the outputs for fwd and bwd directions (in the third dimension).
            outputs_concat =  tf.concat(outputs, axis=2)

            # TODO(we): Add a dense layer (without activation) into num_tags classes and
            # store result in `output_layer`.
            dense = tf.layers.dense(outputs_concat, num_languages, activation=None)
            output_layer = tf.reduce_mean(dense, 1)

            self.predictions = tf.argmax(output_layer, axis=1)

            
            # TODO(we): Generate `weights` as a 1./0. mask of valid/invalid words (using `tf.sequence_mask`).
            # weights = tf.sequence_mask(lengths=self.sentence_lens, dtype=tf.float32)
            

            # Training

            # TODO(we): Define `loss` using `tf.losses.sparse_softmax_cross_entropy`, but additionally
            # use `weights` parameter to mask-out invalid words.
            loss = tf.losses.sparse_softmax_cross_entropy(labels=self.languages, logits=output_layer)

            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")



            # Summaries
            self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(self.languages, self.predictions)
            self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=tf.size(self.sentence_lens))
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
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
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, languages = \
                train.next_batch(batch_size)
            self.session.run(self.reset_metrics)
            self.session.run([self.training, self.summaries["train"]],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                              self.word_ids: word_ids, self.charseq_ids: charseq_ids,
                              self.languages: languages})

    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, languages = \
                dataset.next_batch(batch_size)
            self.session.run([self.update_accuracy, self.update_loss],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                              self.word_ids: word_ids, self.charseq_ids: charseq_ids,
                              self.languages: languages})

        return self.session.run([self.current_accuracy, self.summaries[dataset_name]])[0]

    def predict(self, dataset, batch_size):
        languages = []
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, _ = \
                dataset.next_batch(batch_size)
            languages.extend(self.session.run(self.predictions,
                                              {self.sentence_lens: sentence_lens,
                                               self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                                               self.word_ids: word_ids, self.charseq_ids: charseq_ids}))

        return languages


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
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
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
    train = nli_dataset.NLIDataset("nli-train.txt")
    dev = nli_dataset.NLIDataset("nli-dev.txt", train=train, shuffle_batches=False)
    test = nli_dataset.NLIDataset("nli-test.txt", train=train, shuffle_batches=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, len(train.vocabulary("words")), len(train.vocabulary("chars")), len(train.vocabulary("languages")))

    # Train
    for i in range(args.epochs):
        network.train_epoch(train, args.batch_size)

        network.evaluate("dev", dev, args.batch_size)

    # Predict test data
    with open("{}/nli_test.txt".format(args.logdir), "w", encoding="utf-8") as test_file:
        languages = network.predict(test, args.batch_size)
        for language in languages:
            print(test.vocabulary("languages")[language], file=test_file)
