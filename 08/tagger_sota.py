#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import random
import morpho_dataset
from collections import defaultdict

class MorphoAnalyzer:
    """ Loader for data of morphological analyzer.

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
            self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens")
            self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")
            self.charseqs = tf.placeholder(tf.int32, [None, None], name="charseqs")
            self.charseq_lens = tf.placeholder(tf.int32, [None], name="charseq_lens")
            self.charseq_ids = tf.placeholder(tf.int32, [None, None], name="charseq_ids")
            self.tags = tf.placeholder(tf.int32, [None, None], name="tags")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # TODO(we): Choose RNN cell class according to args.rnn_cell (LSTM and GRU
            # should be supported, using tf.nn.rnn_cell.{BasicLSTM,GRU}Cell).
            num_units = args.rnn_cell_dim
            if args.rnn_cell == 'RNN':
                cell_fw = tf.nn.rnn_cell.BasicRNNCell(num_units)
                cell_bw = tf.nn.rnn_cell.BasicRNNCell(num_units)

            elif args.rnn_cell == 'LSTM':
                cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units)
                cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units)

            else: # Problem when gru selected again. why only 40% acc
                # ?????????
                cell_fw = tf.nn.rnn_cell.GRUCell(num_units)
                cell_bw = tf.nn.rnn_cell.GRUCell(num_units)            

            # Add dropout
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob=1-args.dropout, output_keep_prob=1-args.dropout)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, input_keep_prob=1-args.dropout, output_keep_prob=1-args.dropout)
            # Create word embeddings for num_words of dimensionality args.we_dim
            # using `tf.get_variable`.
            word_embeddings = tf.get_variable('word_embeddings', [num_words, args.we_dim])

            # Embed self.word_ids according to the word embeddings, by utilizing
            # `tf.nn.embedding_lookup`.
            embedded_words = tf.nn.embedding_lookup(word_embeddings, self.word_ids) # which word ids?

            # Convolutional word embeddings (CNNE)

            # Generate character embeddings for num_chars of dimensionality args.cle_dim.
            char_embeddings = tf.get_variable('char_embeddings', [num_chars, args.cle_dim])

            # Embed self.charseqs (list of unique words in the batch) using the character embeddings.
            embedded_chars = tf.nn.embedding_lookup(char_embeddings, self.charseqs)
            #print(embedded_chars)
            # TODO: For kernel sizes of {2..args.cnne_max}, do the following:
            # - use `tf.layers.conv1d` on input embedded characters, with given kernel size
            #   and `args.cnne_filters`; use `VALID` padding, stride 1 and no activation.
            # - perform channel-wise max-pooling over the whole word, generating output
            #   of size `args.cnne_filters` for every word.
            #cnn_filter_no = 0
            outputs = []

            # uncomment to manually to 1d conv
            #embedded_chars_ = tf.expand_dims(embedded_chars, axis=1) # change to shape [n, 1, max_len, dim], so its like an image of height one
            #print('expanded in', embedded_chars_)
            for kernel_size in range(2, args.cnne_max + 1):
                # Manual 1d conv
                #filter_ = tf.get_variable('conv_filter'+str(kernel_size), shape=[1, kernel_size, args.cle_dim, args.cnne_filters])
                #output = tf.nn.conv2d(embedded_chars_, filter_, strides=[1,1,1,1], padding='VALID', name='cnne_layer_'+str(kernel_size))
                #output = tf.squeeze(output, axis=1) # remove extra dim
                #print(output)

                #output = tf.layers.conv1d(embedded_chars, args.cnne_filters, kernel_size, strides=1, padding='VALID', name='cnne_layer_'+str(kernel_size))
                output = tf.layers.conv1d(embedded_chars, args.cnne_filters, kernel_size, strides=1, padding='VALID', activation=None, use_bias=False, name='cnne_layer_'+str(kernel_size))

                # Apply batch norm
                if args.bn:
                    output = tf.layers.batch_normalization(output, training=self.is_training, name='cnn_layer_BN_'+str(kernel_size))
                #output = tf.nn.relu(output, name='cnn_layer_relu_'+str(kernel_size))
                pooling = tf.reduce_max(output, axis=1)

                #print(pooling)
                #cnn_layer_no += 1
                outputs.append(pooling)


            # Concatenate the computed features (in the order of kernel sizes 2..args.cnne_max).
            # Consequently, each word from `self.charseqs` is represented using convolutional embedding
            # (CNNE) of size `(args.cnne_max-1)*args.cnne_filters`.
            concat_output = tf.concat(outputs, axis=-1)
            #print(concat_output)

            # Generate CNNEs of all words in the batch by indexing the just computed embeddings
            # by self.charseq_ids (using tf.nn.embedding_lookup).
            cnne = tf.nn.embedding_lookup(concat_output, self.charseq_ids)
            #print('cnne', cnne)

            # Concatenate the word embeddings (computed above) and the CNNE (in this order).
            embedded_inputs = tf.concat([embedded_words, cnne], axis=-1)
            #print('emb in', embedded_inputs)

            # Using tf.nn.bidirectional_dynamic_rnn, process the embedded inputs.
            # Use given rnn_cell (different for fwd and bwd direction) and self.sentence_lens.
            outputs = embedded_inputs
            for i in range(1, args.layers):  # add more layers (optional)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=outputs, sequence_length=self.sentence_lens, dtype=tf.float32)

            #output1 = tf.nn.batch_normalization(outputs[0], training=self.is_training, name='birnn_bn1'+str(kernel_size))
            #output1 = tf.nn.relu(output1, name='birnn_relu1'+str(kernel_size))
            #output2 = tf.nn.batch_normalization(outputs[0], training=self.is_training, name='birnn_bn2'+str(kernel_size))
            #output2 = tf.nn.relu(output2, name='birnn_relu2'+str(kernel_size))

            # Concatenate the outputs for fwd and bwd directions (in the third dimension).
            output = tf.concat(outputs, axis=-1)

            # Add a dense layer (without activation) into num_tags classes and
            # store result in `output_layer`.
            output_layer = tf.layers.dense(output, num_tags) 
            #print(output_layer)

            # Generate `self.predictions`.
            self.predictions = tf.argmax(output_layer, axis=-1) # 3rd dim!
            #print(self.predictions

            # Generate `weights` as a 1./0. mask of valid/invalid words (using `tf.sequence_mask`).
            weights = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)

            # Training

            # Define `loss` using `tf.losses.sparse_softmax_cross_entropy`, but additionally
            # use `weights` parameter to mask-out invalid words.
            loss = tf.losses.sparse_softmax_cross_entropy(labels=self.tags, logits=output_layer, weights=weights)
            global_step = tf.train.create_global_step()

            #self.learning_rate = tf.get_variable("learning_rate", dtype=tf.float32, initializer=args.learning_rate) 
            #self.learning_rate = tf.Print(self.learning_rate, [self.learning_rate], message='learning rate=')
            # Set adaptable learning rate with decay
            learning_rate = args.learning_rate  # init rate         
            if args.learning_rate_final and args.epochs > 1:
                # Polynomial decay
                if not args.decay_rate: 
                    decay_rate = (args.learning_rate_final / args.learning_rate)**(1 / (args.epochs - 1))
                    learning_rate = tf.train.polynomial_decay(args.learning_rate, global_step, batches, decay_rate, staircase=True) # change lr each batch
                # Exponential decay
                else:
                    decay_rate = args.decay_rate
                    learning_rate = tf.train.exponential_decay(args.learning_rate, global_step, batches, decay_rate, staircase=True) # change lr each batch
            
               
                    #self.learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, batches, decay_rate, staircase=False) # change lr each batch
            
            
            
            
            #else:
                #self.learning_rate = args.learning_rate # init rate

            # Choose optimizer                                              
            if args.optimizer == "SGD" and args.momentum:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=args.momentum) 
                self.training = tf.train.GradientDescentOptimizer(learning_rate) 
            elif args.optimizer == "SGD":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate) 
            else:                
                optimizer = tf.train.AdamOptimizer(learning_rate) 


            # Note how instead of `optimizer.minimize` we first get the # gradients using
            # `optimizer.compute_gradients`, then optionally clip them and
            # finally apply then using `optimizer.apply_gradients`.
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            # TODO: Compute norm of gradients using `tf.global_norm` into `gradient_norm`.
            gradient_norm = tf.global_norm(gradients) 
            # TODO: If args.clip_gradient, clip gradients (back into `gradients`) using `tf.clip_by_global_norm`.            
            if args.clip_gradient is not None:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=args.clip_gradient, use_norm=gradient_norm)
            self.training = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)


            # Summaries
            self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(self.tags, self.predictions, weights=weights)
            self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=tf.reduce_sum(weights))
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.update_loss),
                                           tf.contrib.summary.scalar("train/gradient_norm", gradient_norm),
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
            #self.session.run([self.training, self.update_accuracy, self.summaries["train"]],
            self.session.run([self.training, self.summaries["train"]],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                              self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS],
                              self.tags: word_ids[train.TAGS], self.is_training: True})
        #return self.session.run([self.current_accuracy])

    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = dataset.next_batch(batch_size, including_charseqs=True)
            self.session.run([self.update_accuracy, self.update_loss],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                              self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS],
                              self.tags: word_ids[train.TAGS], self.is_training: False})
        return self.session.run([self.current_accuracy, self.summaries[dataset_name]])[0]

    def predict(self, dataset, batch_size):
        tags = []
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = dataset.next_batch(batch_size, including_charseqs=True)
            tags.extend(self.session.run(self.predictions,
                                         {self.sentence_lens: sentence_lens,
                                          self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                                          self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS], 
                                          self.is_training: False}))
        return tags



if __name__ == "__main__":

    import argparse
    import datetime
    import os
    import re


    def find_analysis(form, tag):

        # Get tags for form in analyzer and guesser lists
        dict_tag_list = [analysis.tag for analysis in analyzer_dictionary.get(form)]
        guesser_tag_list = [analysis.tag for analysis in analyzer_guesser.get(form)]
        combined_tag_list = dict_tag_list + guesser_tag_list 
        total_tags = len(combined_tag_list)

        # Keep tag if empty list or tag already in list
        if total_tags == 0 or tag in combined_tag_list:
            return tag

        # Get tag probs
        tag_probs = defaultdict(lambda: 0)
        for tag in combined_tag_list:
            tag_probs[tag] += 1 / total_tags
        return max(tag_probs, key=tag_probs.get)





    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--cle_dim", default=64, type=int, help="Character-level embedding dimension.")
    parser.add_argument("--cnne_filters", default=32, type=int, help="CNN embedding filters per length.")
    parser.add_argument("--optimizer", default="Adam", type=str, help="Optimizer.")    
    parser.add_argument("--cnne_max", default=4, type=int, help="Maximum CNN filter length.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--recodex", default=False, action="store_true", help="ReCodEx mode.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=64, type=int, help="RNN cell dimension.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate.") 
    parser.add_argument("--learning_rate_final", default=None, type=float, help="Final learning rate.")    
    parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
    parser.add_argument("--momentum", default=None, type=float, help="Momentum.")
    parser.add_argument("--dropout", default=0, type=float, help="Dropout rate.")
    parser.add_argument("--bn", default=False, type=bool, help="Batch normalization.")
    parser.add_argument("--clip_gradient", default=None, type=float, help="Norm for gradient clipping.")
    parser.add_argument("--layers", default=1, type=int, help="Number of rnn layers.")
    parser.add_argument("--anal", default=False, type=bool, help="Filter output with analyzer.")
    parser.add_argument("--decay_rate", default=0, type=float, help="Decay rate.")

    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself


    #def find_analysis(form, tag):

        ## Get tags for form in analyzer and guesser lists
        #dict_tag_list  = [analysis.tag for analysis in analyzer_dictionary.get(form)]
        #guesser_tag_list = [analysis.tag for analysis in analyzer_guesser.get(form)]
        #combined_tag_list = dict_tag_list + guesser_tag_list 
        #total_tags = len(combined_tag_list)

        ## Keep tag if empty list or tag already in list
        #if num_tags == 0 or tag in combined_tag_list:
            #return tag

        ## Get tag probs
        #tag_probs = defaultdict(lambda: 0)
        #for tag in combined_tag_list:
            #tag_probs[tag] += 1 / total_tags
        #return max(tag_probs, key=tag_probs)


    # Load the data
    
    # Subset for sampling/testing

    #Train = Morpho_dataset.MorphoDataset("train.txt")
    
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

    # Train
    for i in range(args.epochs):
        #print(i, args.learning_rate)
        #accuracy = network.train_epoch(train, args.batch_size)
        network.train_epoch(train, args.batch_size)
        #print("train acc = {:.2f}".format(100 * accuracy))      
        accuracy = network.evaluate("dev", dev, args.batch_size)
        print("{:.2f}".format(100 * accuracy))

    # Predict test data
    with open("{}/tagger_sota_test.txt".format(args.logdir), "w") as test_file:
        #print(test_file)
        forms = test.factors[test.FORMS].strings
        lemmas = test.factors[test.LEMMAS].strings
        tags = network.predict(test, args.batch_size)
        #forms = dev.factors[dev.FORMS].strings
        #lemmas = dev.factors[dev.LEMMAS].strings        
        #tags = network.predict(dev, args.batch_size)

        for s in range(len(forms)):
            for i in range(len(forms[s])):
                form = forms[s][i]
                lemma = lemmas[s][i]
                tag = test.factors[test.TAGS].words[tags[s][i]]
                #tag = dev.factors[dev.TAGS].words[tags[s][i]]

                # print('candidates', form, lemma, tag)

                # Use analyzer (optional)
                if args.anal:
                    tag = find_analysis(form, tag)


                #print("{}\t_\t{}".format(form, tag), file=test_file)
                print("{}\t{}\t{}".format(form, lemma, tag), file=test_file)

                #print("{}\t_\t{}".format(forms[s][i], test.factors[test.TAGS].words[tags[s][i]]), file=test_file)
            print("", file=test_file)




            #if dict_tag_list or guesser_list:
            #if (lemma, tag) in dict_tag_list or (lemma, tag) in guesser_lemma_tag_pairs:
            #return lemma, tag
            #else:   
            ##dict_lemma_tag_pairs = [(analysis.lemma, analysis.tag) for analysis in analyzer_dictionary.get(form) if analysis.lemma == lemma]
            ##dict_list  = [(analysis.lemma, analysis.tag) for analysis in analyzer_dictionary.get(form)]
            ##guesser_list = [(analysis.lemma, analysis.tag) for analysis in analyzer_guesser.get(form)]                    
            #print('getting new prob distrib for form using analyzer')

            #if dict_tag_list:
            #lemma, tag = random.choice(dict_tag_list)
            #elif guesser_lemma_tag_pairs:
            #lemma, tag = random.choice(guesser_lemma_tag_pairs)      
            #return lemma, tag       
            #dict_lemma_tag_pairs = [(analysis.lemma, analysis.tag) for analysis in analyzer_dictionary.get(form) if analysis.lemma == lemma]
            #guesser_lemma_tag_pairs = [(analysis.lemma, analysis.tag) for analysis in analyzer_guesser.get(form) if analysis.lemma == lemma]

            #if dict_lemma_tag_pairs or guesser_lemma_tag_pairs:
            #if (lemma, tag) in dict_lemma_tag_pairs or (lemma, tag) in guesser_lemma_tag_pairs:
            #return lemma, tag
            #else:   
            ##dict_lemma_tag_pairs = [(analysis.lemma, analysis.tag) for analysis in analyzer_dictionary.get(form) if analysis.lemma == lemma]
            ##dict_list  = [(analysis.lemma, analysis.tag) for analysis in analyzer_dictionary.get(form)]
            ##guesser_list = [(analysis.lemma, analysis.tag) for analysis in analyzer_guesser.get(form)]                    
            #print('choosing randomly')

            #if dict_lemma_tag_pairs:
            #lemma, tag = random.choice(dict_lemma_tag_pairs)
            #elif guesser_lemma_tag_pairs:
            #lemma, tag = random.choice(guesser_lemma_tag_pairs)      
            #return lemma, tag
