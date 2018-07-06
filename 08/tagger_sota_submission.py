#!/usr/bin/env python3
import sys

#sys.path.append('/home/liefe/py/we')
import numpy as np
import tensorflow as tf
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
            
            # TODO: Training.
            # Define:
            # - loss in `loss`
            # - training in `self.training`
            # - predictions in `self.predictions`
            # - weights in `weights`
            num_units = args.rnn_cell_dim
            if args.rnn_cell == 'RNN':
                cell_fw = tf.nn.rnn_cell.BasicRNNCell(num_units)
                cell_bw = tf.nn.rnn_cell.BasicRNNCell(num_units)

            elif args.rnn_cell == 'GRU':
                cell_fw = tf.nn.rnn_cell.GRUCell(num_units)
                cell_bw = tf.nn.rnn_cell.GRUCell(num_units)       
            
            else: # LSTM default
                cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units)
                cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units)
 
            # Create word embeddings (WE) for num_words of dimensionality args.we_dim
            # using `tf.get_variable`.
            if args.use_wv:
                print('Addding pretrained we to graph', args.use_wv)
                self.word_embeddings = tf.Variable(tf.zeros([vocab_size, args.we_dim], tf.float32))
            
            else: 
                self.word_embeddings = tf.get_variable('word_embeddings', [num_words, args.we_dim])            
                     
            
            # Embed self.word_ids according to the word embeddings, by utilizing
            # `tf.nn.embedding_lookup`.
            embedded_words = tf.nn.embedding_lookup(self.word_embeddings, self.word_ids) # which word ids?
            
            if args.bn_we:
                embedded_words = tf.layers.batch_normalization(embedded_words, training=self.is_training, name='we_bn')
                   
            
            # Convolutional word embeddings (CNNE)
            # Generate character embeddings for num_chars of dimensionality args.cle_dim.
            char_embeddings = tf.get_variable('char_embeddings', [num_chars, args.cle_dim])
            embedded_chars = tf.nn.embedding_lookup(char_embeddings, self.charseqs)
            
            if args.bn_c_1:
                embedded_chars = tf.layers.batch_normalization(embedded_chars, training=self.is_training, name='we_cle')
                   
             
            # TODO: For kernel sizes of {2..args.cnne_max}, do the following:
            # - use `tf.layers.conv1d` on input embedded characters, with given kernel size
            #   and `args.cnne_filters`; use `VALID` padding, stride 1 and no activation.
            # - perform channel-wise max-pooling over the whole word, generating output
            #   of size `args.cnne_filters` for every word.
            features = []  
            for kernel_size in range(2, args.cnne_max + 1):
                conv = tf.layers.conv1d(inputs=embedded_chars, filters=args.cnne_filters, kernel_size=kernel_size,
                                            strides=1, padding='valid', activation=None)       # valid=only fully inside text       
                # Apply batch norm
                if args.bn_c_2:
                    conv = tf.layers.batch_normalization(conv, training=self.is_training, name='cnn_layer_BN_'+str(kernel_size))
                pooling = tf.reduce_max(conv, axis=1)
                act = tf.nn.relu(pooling)
                features.append(act)
            
            # Concatenate the computed features (in the order of kernel sizes 2..args.cnne_max).
            # Consequently, each word from `self.charseqs` is represented using convolutional embedding
            # (CNNE) of size `(args.cnne_max-1)*args.cnne_filters`.
            concat_features = tf.concat(features, axis=1)   
            # Generate CNNEs of all words in the batch by indexing the just computed embeddings
            # by self.charseq_ids (using tf.nn.embedding_lookup).
            embedded_chars = tf.nn.embedding_lookup(concat_features, self.charseq_ids)  
            
            if args.bn_c_3:
                embedded_chars = tf.layers.batch_normalization(embedded_chars, training=self.is_training, name='cnne_bn')
             
            # Concatenate the word embeddings (computed above) and the CNNE (in this order).
            embedded_inputs = tf.concat([embedded_words, embedded_chars], axis=2)            
            
            # Using tf.nn.bidirectional_dynamic_rnn, process the embedded inputs.
            # Use given rnn_cell (different for fwd and bwd direction) and self.sentence_lens.
            outputs = embedded_inputs
            
            if args.bn_in:
                outputs = tf.layers.batch_normalization(outputs, training=self.is_training, name='concat_bn')
                     
            # Add dropout wrapper 
            if args.dropout:
                cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_size=outputs.get_shape()[-1] if args.layers == 1 else tf.TensorShape(num_units), input_keep_prob=1-args.dropout, output_keep_prob=1-args.dropout, variational_recurrent=True, dtype=tf.float32)
                cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, input_size=outputs.get_shape()[-1] if args.layers == 1 else tf.TensorShape(num_units), input_keep_prob=1-args.dropout, output_keep_prob=1-args.dropout, variational_recurrent=True, dtype=tf.float32)            
           
            for i in range(0, args.layers):  # add more layers (optional)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=outputs, 
                                                             sequence_length=self.sentence_lens, dtype=tf.float32)
                
            # Concatenate the outputs for fwd and bwd directions (in the third dimension).
            output = tf.concat(outputs, axis=2)
            
            if args.bn_out:
                output = tf.layers.batch_normalization(output, training=self.is_training, name='out_bn')
                     
            # Add a dense layer (without activation) into num_tags classes and
            # store result in `output_layer`.
            logits = tf.layers.dense(output, num_tags) 

            # Generate `self.predictions`.
            self.predictions = tf.argmax(logits, axis=2) # 3rd dim! 
            
            # Generate `weights` as a 1./0. mask of valid/invalid words (using `tf.sequence_mask`).
            weights = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)
            
            # Training
            
            # L2 regularization
            #l2 = 0
            #if args.l2:
                #print('performing L2 normalization')
                #tv = tf.trainable_variables()
                #l2 = args.l2 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv 
                                                     #if not('bias' in v.name or 'Bias' in v.name or 'noreg' in v.name)]) 
            
            #loss = tf.losses.sparse_softmax_cross_entropy(labels=self.tags, logits=logits, weights=weights) + l2
            
            loss = tf.losses.sparse_softmax_cross_entropy(labels=self.tags, logits=logits, weights=weights)            
            global_step = tf.train.create_global_step()
            
            # For adaptable learning rate if desired
            learning_rate = args.learning_rate  # init rate         
            if args.learning_rate_final and args.epochs > 1:
            # Polynomial decay
                if not args.decay_rate: 
                    decay_rate = (args.learning_rate_final / args.learning_rate)**(1 / (args.epochs - 1))
                learning_rate = tf.train.polynomial_decay(args.learning_rate, global_step, batches_per_epoch, decay_rate, staircase=True) # change lr each batch
            # Exponential decay
            else:
                learning_rate = tf.train.exponential_decay(args.learning_rate, global_step, batches_per_epoch, args.decay_rate, staircase=True) # change lr each batch
        
        
            # Choose optimizer                                              
            if args.optimizer == "SGD" and args.momentum:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=args.momentum) 
                self.training = tf.train.GradientDescentOptimizer(learning_rate) 
            elif args.optimizer == "SGD":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate) 
            else:                
                optimizer = tf.train.AdamOptimizer(learning_rate) 
        
            self.training = optimizer.minimize(loss, global_step=global_step, name="training")
        
            ## Note how instead of `optimizer.minimize` we first get the # gradients using
            ## `optimizer.compute_gradients`, then optionally clip them and
            ## finally apply then using `optimizer.apply_gradients`.
            #gradients, variables = zip(*optimizer.compute_gradients(loss))
            ## TODO: Compute norm of gradients using `tf.global_norm` into `gradient_norm`.
            #gradient_norm = tf.global_norm(gradients) 
            ## TODO: If args.clip_gradient, clip gradients (back into `gradients`) using `tf.clip_by_global_norm`.            
            #if args.clip_gradient is not None:
                #gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=args.clip_gradient, use_norm=gradient_norm)
            #self.training = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)       
      
            
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
                              self.tags: word_ids[train.TAGS], self.is_training: True}) 

    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = dataset.next_batch(batch_size, including_charseqs=True)
            self.session.run([self.update_accuracy, self.update_loss],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                              self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS],
                              self.tags: word_ids[train.TAGS], self.is_training: False }) 
            
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
    import sys
    from collections import defaultdict
      
    def find_analysis_tag(form, tag):

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
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--cle_dim", default=256, type=int, help="Character-level embedding dimension.")
    parser.add_argument("--cnne_filters", default=64, type=int, help="CNN embedding filters per length.")
    parser.add_argument("--optimizer", default="Adam", type=str, help="Optimizer.")    
    parser.add_argument("--cnne_max", default=8, type=int, help="Maximum CNN filter length.")
    parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
    parser.add_argument("--recodex", default=False, action="store_true", help="ReCodEx mode.")
    parser.add_argument("--rnn_cell", default="GRU", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=64, type=int, help="RNN cell dimension.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate.") 
    parser.add_argument("--learning_rate_final", default=None, type=float, help="Final learning rate.")    
    parser.add_argument("--we_dim", default=256, type=int, help="Word embedding dimension.")
    parser.add_argument("--momentum", default=None, type=float, help="Momentum.")
    parser.add_argument("--dropout", default=.5, type=float, help="Dropout rate.")
    parser.add_argument("--bn_we", default=False, type=bool, help="Batch normalization.")
    parser.add_argument("--bn_c_1", default=False, type=bool, help="Batch normalization.")
    parser.add_argument("--bn_c_2", default=False, type=bool, help="Batch normalization.")
    parser.add_argument("--bn_c_3", default=False, type=bool, help="Batch normalization.")
    parser.add_argument("--bn_in", default=False, type=bool, help="Batch normalization.")
    parser.add_argument("--bn_out", default=False, type=bool, help="Batch normalization.")    
    parser.add_argument("--clip_gradient", default=None, type=float, help="Norm for gradient clipping.")
    parser.add_argument("--layers", default=1, type=int, help="Number of rnn layers.")
    parser.add_argument("--anal", default=True, type=bool, help="Filter output with analyzer.")
    parser.add_argument("--decay_rate", default=0, type=float, help="Decay rate.")
    parser.add_argument("--use_wv", default=False, type=bool, help="Use pretrained word embeddings from file")
    parser.add_argument("--l2", default=0, type=float, help="Use l2 regularization.")

    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    train = morpho_dataset.MorphoDataset('czech-pdt-train.txt')
    dev = morpho_dataset.MorphoDataset('czech-pdt-dev.txt', train=train, shuffle_batches=False)
    test = morpho_dataset.MorphoDataset('czech-pdt-test.txt', train=train, shuffle_batches=False)
  
    # These are args I used - all preset as defaults
  
  
    # Data stats
    vocab_size = len(train.factors[train.FORMS].words)
    batches_per_epoch = len(train.sentence_lens) // args.batch_size
    print('num training sents', len(train.sentence_lens))
    print('num batches per epoch', batches_per_epoch)
    print('vocab size = {}, using we_dim = {}'.format(vocab_size, args.we_dim))
    
    analyzer_dictionary = MorphoAnalyzer("czech-pdt-analysis-dictionary.txt")
    analyzer_guesser = MorphoAnalyzer("czech-pdt-analysis-guesser.txt")

        
    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, len(train.factors[train.FORMS].words), len(train.factors[train.FORMS].alphabet),
                      len(train.factors[train.TAGS].words))

    
    # Set we var
    print('Using pretrained embeddings: ', args.use_wv)    
    if args.use_wv:
       
        file = home + '/we/word2vec_cs' + str(args.we_dim) + '.txt_embedded.npy'
        print("Loading pretrained word2vec embeddings from file {}\n".format(file))
        we = np.load(file) # we matrix
        network.session.run(network.word_embeddings.assign(we))
    

    # Train
    for i in range(args.epochs):
        network.train_epoch(train, args.batch_size)
        accuracy = network.evaluate("dev", dev, args.batch_size)
        print("{:.2f}".format(100 * accuracy))
        

    # Predict dev data
    with open("{}/tagger_sota_dev.txt".format(args.logdir), "w") as test_file:
 
        forms = dev.factors[dev.FORMS].strings
        #lemmas = dev.factors[dev.LEMMAS].strings        
        tags = network.predict(dev, args.batch_size)
        print('forms, tags', len(forms), len(tags))

        for s in range(len(forms)):
            for i in range(len(forms[s])):
                form = forms[s][i]
                tag = dev.factors[dev.TAGS].words[tags[s][i]]
                
                # Use analyzer (optional)
                if args.anal:
                    tag = find_analysis_tag(form, tag)

                print("{}\t_\t{}".format(forms[s][i], test.factors[dev.TAGS].words[tags[s][i]]), file=test_file)
            print("", file=test_file)
            
    # Predict test data
    with open("{}/tagger_sota_test.txt".format(args.logdir), "w") as test_file:
        
        forms = test.factors[test.FORMS].strings
        tags = network.predict(test, args.batch_size)

        for s in range(len(forms)):
            for i in range(len(forms[s])):
                form = forms[s][i]
                tag = test.factors[test.TAGS].words[tags[s][i]]

                # Use analyzer (optional)
                if args.anal:
                    tag = find_analysis_tag(form, tag)

                print("{}\t_\t{}".format(forms[s][i], test.factors[dev.TAGS].words[tags[s][i]]), file=test_file)
            print("", file=test_file)    