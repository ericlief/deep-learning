#!/usr/bin/env python3
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

    def construct(self, args, num_words, num_chars, num_langs):
        with self.session.graph.as_default():
            # Inputs
            self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens") #???  don't have
            self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")
            self.charseqs = tf.placeholder(tf.int32, [None, None], name="charseqs")
            self.charseq_lens = tf.placeholder(tf.int32, [None], name="charseq_lens")
            self.charseq_ids = tf.placeholder(tf.int32, [None, None], name="charseq_ids")
            self.tags = tf.placeholder(tf.int32, [None, None], name="tags")            
            self.languages = tf.placeholder(tf.int32, [None], name="languages")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")
            
            # Convert to one-hot repr
            labels = tf.one_hot(self.languages, num_langs)
            
            # TODO: Training.
            # Define:
            # - loss in `loss`
            # - training in `self.training`
            # - predictions in `self.predictions`
                        
            # Use pretrained embeddings or create word embeddings (WE) for num_words of dimensionality args.we_dim
            # using `tf.get_variable`. Then embed self.word_ids according to the word embeddings, by utilizing
      
            if args.use_wv:
                print('addding pretrained we to graph', args.use_wv)
                #word_embeddings = tf.get_variable('word_embeddings', shape=wv.shape, initializer=tf.constant_initializer(wv), trainable=False)
                #word_embeddings = tf.Variable(tf.random_uniform([vocab_size, args.we_dim], -1.0, 1.0, name='word_embeddings'))
                self.word_embeddings = tf.Variable(tf.zeros([num_words, args.we_dim], tf.float32))
            
            else: 
                self.word_embeddings = tf.get_variable('word_embeddings', [num_words, args.we_dim])            
            
            embedded_words = tf.nn.embedding_lookup(self.word_embeddings, self.word_ids)  
            

            # Character-level word embeddings (CLE)
            
            # TODO: Generate character embeddings for num_chars of dimensionality args.cle_dim. 
            char_embeddings = tf.get_variable('char_embeddings', [num_chars, args.cle_dim])
            
            # TODO: Embed self.charseqs (list of unique words in the batch) using the character embeddings.
            embedded_chars = tf.nn.embedding_lookup(char_embeddings, self.charseqs)            
            
        
            # TODO: For kernel sizes of {2..args.cnne_max}, do the following:
            # - use `tf.layers.conv1d` on input embedded characters, with given kernel size
            #   and `args.cnne_filters`; use `VALID` padding, stride 1 and no activation.
            # - perform channel-wise max-pooling over the whole word, generating output
            #   of size `args.cnne_filters` for every word.
            features = []  
            for kernel_size in range(2, args.cnne_max + 1):
                #with tf.name_scope("cnne-maxpool-%s" % kernel_size):
                with tf.variable_scope("cnne-maxpool-%s" % kernel_size):
                    
                    conv = tf.layers.conv1d(inputs=embedded_chars, filters=args.cnne_filters, kernel_size=kernel_size,
                                                strides=1, padding='valid', activation=None)       # valid=only fully inside text       
                    pooling = tf.reduce_max(conv, axis=1)
                    features.append(pooling)
                
            # Concatenate the computed features (in the order of kernel sizes 2..args.cnne_max).
            # Consequently, each word from `self.charseqs` is represented using convolutional embedding
            # (CNNE) of size `(args.cnne_max-1)*args.cnne_filters`.
            cnne_features = tf.concat(features, axis=1)   
            # Generate CNNEs of all words in the batch by indexing the just computed embeddings
            # by self.charseq_ids (using tf.nn.embedding_lookup).
            embedded_charseqs = tf.nn.embedding_lookup(cnne_features, self.charseq_ids)  
                


            # Tag embeddings
            tag_embeddings = tf.get_variable('tag_embeddings', [num_words, args.tag_dim])
            embedded_tags = tf.nn.embedding_lookup(tag_embeddings, self.tags) 
 
         
            # Concatenate the word embeddings (computed above), the CNNE, and the tags, along the last dim
            # The inputs are 3D (sent, word, dim)
            print('we, cle, tag dims', embedded_words, embedded_charseqs, embedded_tags) # -> (?,?,128/24/32)
            embedded_inputs = tf.concat([embedded_words, embedded_charseqs, embedded_tags], axis=2)          
            print('embed in', embedded_inputs) # -> (?,?,184)   
            
            with tf.name_scope('text'):
                cell_fw = tf.nn.rnn_cell.GRUCell(args.rnn_dim)
                cell_bw = tf.nn.rnn_cell.GRUCell(args.rnn_dim)             
                
                if args.dropout_text:
                    cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_size=embedded_inputs.get_shape()[-1], input_keep_prob=1-args.dropout_text, output_keep_prob=1-args.dropout_text, variational_recurrent=True, dtype=tf.float32)
                    cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, input_size=embedded_inputs.get_shape()[-1], input_keep_prob=1-args.dropout_text, output_keep_prob=1-args.dropout_text, variational_recurrent=True, dtype=tf.float32)            
                              
                _, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedded_inputs, self.sentence_lens, dtype=tf.float32, scope='text')
                
                sum_states = tf.reduce_sum(states, axis=0)
                print('sum states', sum_states)
                
        
            averaged_states = tf.reduce_mean(sum_states, axis=-1)
            print('ave states', averaged_states)
      
            # Add a dense layer (without activation) into num_languages classes and
            # store result in `logits`.
            #logits = tf.layers.dense(text_features, num_langs) 
            #print('logits', logits) # -> (?,?,11) ~ (?, 11)
            
            logits = tf.layers.dense(sum_states, num_langs) 
            print('logits', logits) # -> (?,?,11) ~ (?, 11)
            
            
            #print('labels', self.languages, self.languages.get_shape().as_list()) # ->(?,?), (None,None)
            #logits = tf.layers.flatten(logits)
            #print('flat out', logits)
            #self.languages = tf.reshape(self.languages, [-1,1])
            #print('reshaped labels', self.languages)

            # Generate `self.predictions`.
            self.predictions = tf.argmax(logits, axis=-1) # 3rd dim!
            # Convert predictions into one-hot
            self.predictions = tf.one_hot(self.predictions, num_langs)
            #print(self.predictions)     # -> (?,11)
            
            # Generate `weights` as a 1./0. mask of valid/invalid words (using `tf.sequence_mask`).
            #weights = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)
            
            # Training
            #loss = tf.losses.sparse_softmax_cross_entropy(labels=self.languages, logits=logits)
            loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
            global_step = tf.train.create_global_step()
            
            # For adaptable learning rate if desired
            #self.learning_rate = tf.get_variable("learning_rate", dtype=tf.float32, initializer=args.learning_rate) 
            # Set adaptable learning rate with decay
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
            #self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(labels_hot, self.predictions)
            self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(labels, self.predictions)
            self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=tf.size(self.sentence_lens))
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
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, languages = \
                train.next_batch(batch_size)
            #languages = np.reshape(languages, [-1,1])
            #print(languages, languages.shape, charseq_ids.shape, word_ids.shape)
            self.session.run(self.reset_metrics)
            self.session.run([self.training, self.summaries["train"]],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                              self.word_ids: word_ids, self.charseq_ids: charseq_ids,
                              self.tags: tags, 
                              self.languages: languages, self.is_training: True})

    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, languages = \
                dataset.next_batch(batch_size)
            #languages = np.reshape(languages, [-1,1])            
            self.session.run([self.update_accuracy, self.update_loss],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                              self.word_ids: word_ids, self.charseq_ids: charseq_ids,
                              self.tags: tags,                               
                              self.languages: languages, self.is_training: False })

        return self.session.run([self.current_accuracy, self.summaries[dataset_name]])[0]

    def predict(self, dataset, batch_size):
        languages = []
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, _ = \
                dataset.next_batch(batch_size)
            predictions = self.session.run(self.predictions,
                                              {self.sentence_lens: sentence_lens,
                                               self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                                               self.word_ids: word_ids, self.charseq_ids: charseq_ids,
                                               self.tags: tags, self.is_training: False})
            #print(np.argmax(predictions, axis=1))
            #print(list(np.argmax(predictions, axis=1)))
            languages.extend(list(np.argmax(predictions, axis=1)))
            #languages.extend(list(np.reshape(np.argmax(predictions, axis=1)), [-1]))
            #languages.extend(predictions)
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
    parser.add_argument("--we_dim", default=32, type=int, help="Word embedding dimension.")
    parser.add_argument("--cle_dim", default=16, type=int, help="Character-level embedding dimension.")
    parser.add_argument("--tag_dim", default=16, type=int, help="Tag embedding dimension.")    
    parser.add_argument("--rnn_dim", default=32, type=int, help="Maximum CNN filter length.")
    parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
    parser.add_argument("--cnne_filters", default=8, type=int, help="CNN embedding filters per length.")
    parser.add_argument("--cnne_max", default=4, type=int, help="Maximum CNN filter length.")    
    parser.add_argument("--dropout_char", default=0, type=float, help="Dropout rate.")    
    parser.add_argument("--dropout_word", default=0, type=float, help="Dropout rate.")
    parser.add_argument("--dropout_text", default=0, type=float, help="Dropout rate.")    
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--bn", default=False, type=bool, help="Batch normalization.")
    parser.add_argument("--optimizer", default="Adam", type=str, help="Optimizer.")    
    parser.add_argument("--clip_gradient", default=None, type=float, help="Norm for gradient clipping.")
    parser.add_argument("--layers", default=1, type=int, help="Number of rnn layers.")
    parser.add_argument("--decay_rate", default=0, type=float, help="Decay rate.")
    parser.add_argument("--use_wv", default=False, type=bool, help="Use pretrained word embeddings.")
    parser.add_argument("--recodex", default=False, action="store_true", help="ReCodEx mode.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate.") 
    parser.add_argument("--learning_rate_final", default=None, type=float, help="Final learning rate.")    
    parser.add_argument("--momentum", default=None, type=float, help="Momentum.")
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
    
    # Data stats
    num_words, num_chars, num_tags, num_langs, num_sents = len(train.vocabulary("words")), len(train.vocabulary("chars")), len(train.vocabulary("tags")), len(train.vocabulary("languages")), len(train._sentence_lens)
    batches_per_epoch = num_sents // args.batch_size
    print('num training sents', num_sents)
    print('num batches per epoch', batches_per_epoch)
    print('num words = {}, num chars = {}, num tags = {}, num langs = {}, using we_dim = {}'.format(num_words, num_chars, num_tags, num_langs, args.we_dim))
    print('we dim = {}, cle = {}, tag dim = {}'.format(args.we_dim, args.cle_dim, args.tag_dim))
      
    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, len(train.vocabulary("words")), len(train.vocabulary("chars")), len(train.vocabulary("languages")))

    # Train
    for i in range(args.epochs):
        network.train_epoch(train, args.batch_size)

        accuracy = network.evaluate("dev", dev, args.batch_size)
        print("{:.2f}".format(100 * accuracy))
        
    # Predict dev data
    with open("{}/nli_dev.txt".format(args.logdir), "w", encoding="utf-8") as test_file:
        languages = network.predict(dev, args.batch_size)
        for language in languages:
            print(test.vocabulary("languages")[language], file=test_file)


    # Predict test data
    with open("{}/nli_test.txt".format(args.logdir), "w", encoding="utf-8") as test_file:
        languages = network.predict(test, args.batch_size)
        for language in languages:
            print(test.vocabulary("languages")[language], file=test_file)
