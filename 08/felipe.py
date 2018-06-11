# coding=utf-8

#Team: Felipe Vianna and Yuu Sakagushi
# Felipe Vianna: 72ef319b-1ef9-11e8-9de3-00505601122b
# Yuu Sakagushi: d9fbf49b-1c71-11e8-9de3-00505601122b


import numpy as np
import tensorflow as tf

import morpho_dataset

class MorphoAnalyzer:
    """Loader for data of morphological analyzer.
    The loaded analyzer provides an only method `get(word)` returning
    a list of analyses, each containing two fields `lemma` and `tag`.
    If an analysis of the word is not found, an empty list is returned.
    """

    class LemmaTag:
        def __init__(self, lemma, tag):
            self.lemma = lemma
            self.tag = tag

    def __init__(self, filename):
        self.analyses = {}

        with open(filename, "r", encoding="utf-8") as analyzer_file:
            for line in analyzer_file:
                line = line.rstrip("\n")
                columns = line.split("\t")

                analyses = []
                for i in range(1, len(columns) - 1, 2):
                    analyses.append(MorphoAnalyzer.LemmaTag(columns[i], columns[i + 1]))
                self.analyses[columns[0]] = analyses

    def get(self, word):
        return self.analyses.get(word, [])


class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, num_words, num_chars, num_tags):
        with self.session.graph.as_default():
            # Inputs
               # Inputs
            self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens")
            self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")
            self.charseqs = tf.placeholder(tf.int32, [None, None], name="charseqs")
            self.charseq_lens = tf.placeholder(tf.int32, [None], name="charseq_lens")
            self.charseq_ids = tf.placeholder(tf.int32, [None, None], name="charseq_ids")
            self.tags = tf.placeholder(tf.int32, [None, None], name="tags")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # TODO: Training.
            # Define:
            # - loss in `loss`
            # - training in `self.training`
            # - predictions in `self.predictions`
            # - weights in `weights`

            rnn_cell_dim = 140
            we_dim = 140
            cle_dim = 70
            cnne_filters = 25
            cnne_max = 7
            
            fwd = tf.nn.rnn_cell.GRUCell(rnn_cell_dim)
            bwd = tf.nn.rnn_cell.GRUCell(rnn_cell_dim)

            we = tf.get_variable("var", shape=[num_words, we_dim])
            embeded_we = tf.nn.embedding_lookup(params=we, ids=self.word_ids)


            # Convolutional word embeddings (CNNE)
            cnne = tf.get_variable("var_cle", shape=[num_chars, cle_dim])
            embeded_cle = tf.nn.embedding_lookup(params=cnne, ids=self.charseqs)

            features=[]
            for k in range(2, cnne_max+1):
                conv = tf.layers.conv1d(inputs=embeded_cle, filters=cnne_filters, kernel_size=k,
                                            strides=1, padding='valid', activation=None)

                pool = tf.reduce_max(input_tensor=conv, axis=1)
                features.append(pool)

            concat_cnne = tf.concat(features, axis=1)
            embeded_cnne = tf.nn.embedding_lookup(params=concat_cnne, ids=self.charseq_ids)

            embeded = tf.concat([embeded_we, embeded_cnne], axis=2)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwd, cell_bw=bwd, inputs=embeded,
                                                    sequence_length=self.sentence_lens, dtype=tf.float32)

            outputs_concat =  tf.concat(outputs, axis=2)
            output_layer = tf.layers.dense(outputs_concat, num_tags, activation=None)
            self.predictions = tf.argmax(output_layer, axis=2)
            weights = tf.sequence_mask(lengths=self.sentence_lens, dtype=tf.float32)


            # Training
            loss = tf.losses.sparse_softmax_cross_entropy(labels=self.tags, logits=output_layer, weights=weights)
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")


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
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = train.next_batch(batch_size, including_charseqs=True)
            self.session.run(self.reset_metrics)
            self.session.run([self.training, self.summaries["train"]],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                              self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS],
                              self.tags: word_ids[train.TAGS]})

    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = dataset.next_batch(batch_size, including_charseqs=True)
            self.session.run([self.update_accuracy, self.update_loss],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                              self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS],
                              self.tags: word_ids[train.TAGS]})
        return self.session.run([self.current_accuracy, self.summaries[dataset_name]])[0]

    def predict(self, dataset, batch_size):
        tags = []
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = dataset.next_batch(batch_size, including_charseqs=True)
            tags.extend(self.session.run(self.predictions,
                                         {self.sentence_lens: sentence_lens,
                                          self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                                          self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS]}))
        return tags


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
    parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    train = morpho_dataset.MorphoDataset("czech-pdt-train.txt")
    dev = morpho_dataset.MorphoDataset("czech-pdt-dev.txt", train=train, shuffle_batches=False)
    test = morpho_dataset.MorphoDataset("czech-pdt-test.txt", train=train, shuffle_batches=False)

    batches = len(train.sentence_lens) // args.batch_size
    print('num sents', len(train.sentence_lens))
    print('num batches', batches)
    
    analyzer_dictionary = MorphoAnalyzer("czech-pdt-analysis-dictionary.txt")
    analyzer_guesser = MorphoAnalyzer("czech-pdt-analysis-guesser.txt")

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, len(train.factors[train.FORMS].words), len(train.factors[train.FORMS].alphabet),
                      len(train.factors[train.TAGS].words))
    ## Load the data
    #train = morpho_dataset.MorphoDataset(\"czech-pdt-train.txt\")
    #dev = morpho_dataset.MorphoDataset(\"czech-pdt-dev.txt\", train=train, shuffle_batches=False)
    #test = morpho_dataset.MorphoDataset(\"czech-pdt-test.txt\", train=train, shuffle_batches=False)

    #analyzer_dictionary = MorphoAnalyzer(\"czech-pdt-analysis-dictionary.txt\")
    #analyzer_guesser = MorphoAnalyzer(\"czech-pdt-analysis-guesser.txt\")

    ## Construct the network
    #network = Network(threads=args.threads)
    #network.construct(args, len(train.factors[train.FORMS].words), len(train.factors[train.FORMS].alphabet),
                      #len(train.factors[train.TAGS].words))

    # Train
    for i in range(args.epochs):
        network.train_epoch(train, args.batch_size)

        network.evaluate("dev", dev, args.batch_size)
      

    # Predict test data
    with open("{}/tagger_sota_test.txt".format(args.logdir), "w", encoding="utf-8") as test_file:
        forms = test.factors[test.FORMS].strings
        tags = network.predict(test, args.batch_size)
        for s in range(len(forms)):
            for i in range(len(forms[s])):
                print("{}\t_\t{}".format(forms[s][i], test.factors[test.TAGS].words[tags[s][i]]), file=test_file)
            print("", file=test_file)
