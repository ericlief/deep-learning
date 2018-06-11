#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import morpho_dataset

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

    def construct(self, args, source_chars, target_chars, bow, eow):
        with self.session.graph.as_default():
            # Inputs
            self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens")
            self.source_ids = tf.placeholder(tf.int32, [None, None], name="source_ids")
            self.source_seqs = tf.placeholder(tf.int32, [None, None], name="source_seqs")
            self.source_seq_lens = tf.placeholder(tf.int32, [None], name="source_seq_lens")
            self.target_ids = tf.placeholder(tf.int32, [None, None], name="target_ids")
            self.target_seqs = tf.placeholder(tf.int32, [None, None], name="target_seqs")
            self.target_seq_lens = tf.placeholder(tf.int32, [None], name="target_seq_lens")

            # TODO: Training. The rest of the code assumes that
            # - when training the decoder, the output layer with logits for each generated
            #   character is in `output_layer` and the corresponding predictions are in
            #   `self.predictions_training`.
            # - the `target_ids` contains the gold generated characters
            # - the `target_lens` contains number of valid characters for each lemma
            # - when running decoder inference, the predictions are in `self.predictions`
            #   and their lengths in `self.prediction_lens`.
            
            # Append EOW after target_seqs
            target_seqs = tf.reverse_sequence(self.target_seqs, self.target_seq_lens, 1)
            target_seqs = tf.pad(target_seqs, [[0, 0], [1, 0]], constant_values=eow)
            target_seq_lens = self.target_seq_lens + 1
            target_seqs = tf.reverse_sequence(target_seqs, target_seq_lens, 1)

             
            # Encoder
            # TODO: Generate source embeddings for source chars, of shape [source_chars, args.char_dim].
            source_embeddings = tf.get_variable('source_embeddings', [source_chars, args.char_dim])
                                            
            # Embed the self.source_seqs using the source embeddings. Only unique words here
            source_encoded = tf.nn.embedding_lookup(source_embeddings, self.source_seqs)
            #print('init cle', source_encoded) # (?,?,64)
            
            # Using a GRU with dimension args.rnn_dim, process the embedded self.source_seqs
            # using bidirectional RNN. Store the summed fwd and bwd outputs in `source_encoded`
            # and the summed fwd and bwd states into `source_states`.
            cell_fw = tf.nn.rnn_cell.GRUCell(args.rnn_dim)
            cell_bw = tf.nn.rnn_cell.GRUCell(args.rnn_dim)             
            source_outputs, source_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, source_encoded, self.source_seq_lens, dtype=tf.float32, scope='encoder')
            
            #print('source outputs', source_outputs, source_states, 'source states', source_states, 'batch size', tf.shape(source_states)[0]) # (?, 200) = encoded seq embedding 
            source_encoded = tf.reduce_sum(source_outputs, axis=0) 
            #print('source output summed', source_encoded)
            source_states = tf.reduce_sum(source_states, axis=0) 
            #print('source states summed', source_encoded)            
            
            # Index the unique words using self.source_ids and self.target_ids.
            sentence_mask = tf.sequence_mask(self.sentence_lens)
            #print('4d src encoded', tf.nn.embedding_lookup(source_encoded, self.source_ids)) # (?,?,?,64)
            source_encoded = tf.boolean_mask(tf.nn.embedding_lookup(source_encoded, self.source_ids), sentence_mask)
            #print('source encoded look up + mask', source_encoded) # (?,?,64)
            source_states = tf.boolean_mask(tf.nn.embedding_lookup(source_states, self.source_ids), sentence_mask)
            #print('source states?', source_states) # (?,?,64) -> (?,64) 
            source_lens = tf.boolean_mask(tf.nn.embedding_lookup(self.source_seq_lens, self.source_ids), sentence_mask)
            
            target_seqs = tf.boolean_mask(tf.nn.embedding_lookup(target_seqs, self.target_ids), sentence_mask)
            target_lens = tf.boolean_mask(tf.nn.embedding_lookup(target_seq_lens, self.target_ids), sentence_mask)

            # Decoder
            # TODO: Generate target embeddings for target chars, of shape [target_chars, args.char_dim].
            target_embeddings = tf.get_variable('tar_embeddings', [target_chars, args.char_dim])

            # Embed the target_seqs using the target embeddings.
            embedded_target = tf.nn.embedding_lookup(target_embeddings, target_seqs)
            #print('target_embed', embedded_target)
            
            # TODO: Generate a decoder GRU with dimension args.rnn_dim.
            decoder_cell = tf.nn.rnn_cell.GRUCell(args.rnn_dim)            
            
            # Create a `decoder_layer` -- a fully connected layer with
            # target_chars neurons used in the decoder to classify into target characters.
            decoder_layer = tf.layers.Dense(target_chars)
            # batch_sz = tf.shape(source_states)[0]   
            
            # Attention
            # Generate three fully connected layers without activations:
            # - `source_layer` with args.rnn_dim units
            # - `state_layer` with args.rnn_dim unitssource_encoded
            # - `weight_layer` with 1 unit
            source_layer = tf.layers.Dense(args.rnn_dim)
            state_layer = tf.layers.Dense(args.rnn_dim)
            weight_layer = tf.layers.Dense(1)            
            def with_attention(inputs, states):
                # Generate the attention

                # Project source_encoded using source_layer.
                #source_layer = tf.layers.dense(source_encoded, args.rnn_dim)
                source_layer_projection = source_layer(source_encoded)
                #print('src layer', source_layer)
                
                # Expand target RNN states s: Change shape of states from [a, b] to [a, 1, b]; mid coord is char
                states = tf.expand_dims(states, axis=1)
                #print('expanded src state', states)
                
                # Project it using state_layer.
                #state_layer = tf.layers.dense(states, args.rnn_dim)
                state_layer_projection = state_layer(states)
                #print('state layer', state_layer)
                
                # Sum the two above projections, apply tf.tanh and project the result using weight_layer.
                # The result has shape [x, y, 1]. 
                score = tf.nn.tanh(source_layer_projection + state_layer_projection) # = e tanh(Vh_j + Ws + b) = (?,?,64) = (w, c, dim)
                #print('score or e <= 1', score)
                # weight_layer = tf.layers.dense(score, 1)  # weight layer with 1 unit 
                weight_layer_projection = weight_layer(score)
                #print('weight layer', weight_layer)
                
                # Apply tf.nn.softmax to the latest result, using axis corresponding to source characters (= prob)
                a = tf.nn.softmax(weight_layer_projection, axis=1)
                #print('a', a)
                
                # Multiply the source_encoded by the latest result, and sum the results with respect
                # to the axis corresponding to source characters. This is the final attention.
                attn = tf.reduce_sum(tf.multiply(source_encoded, a), axis=1)  # weighting 
                #print('attn', attn)
                # TODO: Return concatenation of inputs and the computed attention.
                return tf.concat([inputs, attn], axis=1) 
            
            # The DecoderTraining will be used during training. It will output logits for each
            # target character.    
            class DecoderTraining(tf.contrib.seq2seq.Decoder):
                #batch_sz = tf.shape(source_states)[0]
                #def __init__(self, target_embeddings, batch_size):
                    #self.batch_size = batch_size
                    #self.target_embeddings = target_embeddings
            
                @property
                def batch_size(self): return tf.shape(source_states)[0] # TODO: Return size of the batch, using for example source_states size
                @property
                def output_dtype(self): return tf.float32 # Type for logits of target characters
                @property
                def output_size(self): return target_chars # Length of logits for every output (= num or dim of char map)
            
                def initialize(self, name=None):
                    finished = tf.equal(target_lens, 0) # False if target_lens > 0 else True  
                    states = source_states # Initial decoder state to use
                    # Embed BOW characters of shape [self.batch_size].  
                    inputs = tf.nn.embedding_lookup(target_embeddings, tf.fill([self.batch_size], bow))
                    # Call with_attention on the embedded BOW characters of shape [self.batch_size].
                    inputs = with_attention(inputs, states)
                    #print('init embedded bow inputs', inputs)
                    return finished, inputs, states
            
                def step(self, time, inputs, states, name=None):
                    # Run the decoder GRU cell using inputs and states 
                    outputs, states = decoder_cell(inputs, states) # logits
                    #print('step: outputs from decoder cell', outputs, 'states', states)
                    # Apply the decoder_layer on outputs.       
                    outputs = decoder_layer(outputs)  # now dim = alphabet  
                    #outputs = tf.argmax(outputs, axis=-1, output_type=tf.int32)
                    #print('outputs from decoder layer', outputs)
                    # Next input are words with index `time` in target_embedded.# TODO: 
                    # Embed `outputs` using target_embeddings and pass it to with_attention.
                    next_inputs = embedded_target[:, time]
                    next_inputs = with_attention(next_inputs, states)
                    
                    # TODO: False if target_lens > time + 1, True otherwise.
                    finished = tf.less_equal(target_lens, time + 1) # False if target_lens > time + 1 else True  
                    return outputs, states, next_inputs, finished
            
            output_layer, _, _ = tf.contrib.seq2seq.dynamic_decode(DecoderTraining()) # outputs  
            #print('final outputs', output_layer)
            
            # Predictions find most predicted char in 3D here (axis 2=char dim)
            self.predictions_training = tf.argmax(output_layer, axis=-1, output_type=tf.int32)                
            
            # The DecoderPrediction will be used during prediction. It will
            # directly output the predicted target characters.
            class DecoderPrediction(tf.contrib.seq2seq.Decoder):
                @property
                def batch_size(self): return tf.shape(source_states)[0] # TODO: Return size of the batch, using for example source_states size
                @property
                def output_dtype(self): return tf.int32 # Type for predicted target characters
                @property
                def output_size(self): return 1 # Will return just one output
        
                def initialize(self, name=None):
                    finished = tf.equal(target_lens, 0)
                    states =  source_states
                    # Embed BOW characters of shape [self.batch_size]. You can use tf.fill to generate BOWs of appropriate size.
                    inputs = tf.nn.embedding_lookup(target_embeddings, tf.fill([self.batch_size], bow))
                    # Call with_attention on the embedded BOW characters of shape [self.batch_size].
                    inputs = with_attention(inputs, states)
                    return finished, inputs, states
        
                def step(self, time, inputs, states, name=None):
                    # Run the decoder GRU cell using inputs and states.
                    outputs, states = decoder_cell(inputs, states) # logits
                    # Apply the decoder_layer on outputs.
                    outputs = decoder_layer(outputs) # to char dim, e.g. (?, 91)
                    #print('outputs from decoder', outputs)
                    # Use tf.argmax to choose most probable class (supply parameter `output_type=tf.int32`)
                    outputs = tf.argmax(outputs, axis=1, output_type=tf.int32) # get most prrob char (?,) vector = (?,1)
                    #print('outputs argmax, the class', outputs)
                    # Embed `outputs` using target_embeddings and pass it to with_attention.
                    next_inputs = tf.nn.embedding_lookup(target_embeddings, outputs)
                    #print('outputs argmax', outputs)
                    next_inputs = with_attention(next_inputs, states)
                    # Checkc if fone. True where outputs==eow, False otherwise
                    finished = tf.equal(outputs, eow) # TODO: True where outputs==eow, False otherwise
                    #print('next_in ', next_inputs)
                    
                    return outputs, states, next_inputs, finished
                
            self.predictions, _, self.prediction_lens = tf.contrib.seq2seq.dynamic_decode(
                        DecoderPrediction(), maximum_iterations=tf.reduce_max(source_lens) + 10)            
            #print('pred tens', self.predictions)	
            
            # Training
            weights = tf.sequence_mask(target_lens, dtype=tf.float32)
            loss = tf.losses.sparse_softmax_cross_entropy(target_seqs, output_layer, weights=weights)
            global_step = tf.train.create_global_step()
            
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
    
            
            
            
            #self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

            # Summaries
            accuracy_training = tf.reduce_all(tf.logical_or(
                tf.equal(self.predictions_training, target_seqs),
                tf.logical_not(tf.sequence_mask(target_lens))), axis=1)
            self.current_accuracy_training, self.update_accuracy_training = tf.metrics.mean(accuracy_training)

            minimum_length = tf.minimum(tf.shape(self.predictions)[1], tf.shape(target_seqs)[1])
            accuracy = tf.logical_and(
                tf.equal(self.prediction_lens, target_lens),
                tf.reduce_all(tf.logical_or(
                    tf.equal(self.predictions[:, :minimum_length], target_seqs[:, :minimum_length]),
                    tf.logical_not(tf.sequence_mask(target_lens, maxlen=minimum_length))), axis=1))
            self.current_accuracy, self.update_accuracy = tf.metrics.mean(accuracy)

            self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=tf.reduce_sum(weights))
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.update_loss),
                                           tf.contrib.summary.scalar("train/gradient_norm", gradient_norm),                                           
                                           tf.contrib.summary.scalar("train/accuracy", self.update_accuracy_training)]
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
            sentence_lens, _, charseq_ids, charseqs, charseq_lens = train.next_batch(batch_size, including_charseqs=True)
            self.session.run(self.reset_metrics)
            self.session.run(
                [self.training, self.summaries["train"]],
                {self.sentence_lens: sentence_lens,
                 self.source_ids: charseq_ids[train.FORMS], self.target_ids: charseq_ids[train.LEMMAS],
                 self.source_seqs: charseqs[train.FORMS], self.target_seqs: charseqs[train.LEMMAS],
                 self.source_seq_lens: charseq_lens[train.FORMS], self.target_seq_lens: charseq_lens[train.LEMMAS]})

    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            sentence_lens, _, charseq_ids, charseqs, charseq_lens = dataset.next_batch(batch_size, including_charseqs=True)
            self.session.run([self.update_accuracy, self.update_loss],
                             {self.sentence_lens: sentence_lens,
                              self.source_ids: charseq_ids[train.FORMS], self.target_ids: charseq_ids[train.LEMMAS],
                              self.source_seqs: charseqs[train.FORMS], self.target_seqs: charseqs[train.LEMMAS],
                              self.source_seq_lens: charseq_lens[train.FORMS], self.target_seq_lens: charseq_lens[train.LEMMAS]})
        return self.session.run([self.current_accuracy, self.summaries[dataset_name]])[0]

    def predict(self, dataset, batch_size):
        lemmas = []
        while not dataset.epoch_finished():
            sentence_lens, _, charseq_ids, charseqs, charseq_lens = dataset.next_batch(batch_size, including_charseqs=True)
            predictions, prediction_lengths = self.session.run(
                [self.predictions, self.prediction_lens],
                {self.sentence_lens: sentence_lens, self.source_ids: charseq_ids[train.FORMS],
                 self.source_seqs: charseqs[train.FORMS], self.source_seq_lens: charseq_lens[train.FORMS]})

            for length in sentence_lens:
                lemmas.append([])
                for i in range(length):
                    lemmas[-1].append("")
                    for j in range(prediction_lengths[i] - 1):
                        lemmas[-1][-1] += train.factors[train.LEMMAS].alphabet[predictions[i][j]]
                predictions, prediction_lengths = predictions[length:], prediction_lengths[length:]

        return lemmas


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    def find_analysis_lemma(form, lemma):

        # Get tags for form in analyzer and guesser lists
        dict_lemma_list = [analysis.lemma for analysis in analyzer_dictionary.get(form)]
        guesser_lemma_list = [analysis.lemma for analysis in analyzer_guesser.get(form)]
        combined_lemma_list = dict_lemma_list + guesser_lemma_list 
        #print(combined_lemma_list)
        total_lemmas = len(combined_lemma_list)

        # Keep tag if empty list or tag already in list
        if total_lemmas== 0 or lemma in combined_lemma_list:
            return lemma

        # Get lemma probs
        lemma_probs = defaultdict(lambda: 0)
        for lemma in combined_lemma_list:
            lemma_probs[lemma] += 1 / total_lemmas
        most_freq_lemma = max(lemma_probs, key=lemma_probs.get)
        #print(most_freq_lemma, lemma_probs[most_freq_lemma])
        return most_freq_lemma
    
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--char_dim", default=64, type=int, help="Character-level embedding dimension.")
    parser.add_argument("--cnne_filters", default=32, type=int, help="CNN embedding filters per length.")
    parser.add_argument("--optimizer", default="Adam", type=str, help="Optimizer.")    
    parser.add_argument("--cnne_max", default=4, type=int, help="Maximum CNN filter length.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--recodex", default=False, action="store_true", help="ReCodEx mode.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_dim", default=64, type=int, help="RNN cell dimension.")
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

    # Load the data
    train = morpho_dataset.MorphoDataset("czech-pdt-train.txt")
    dev = morpho_dataset.MorphoDataset("czech-pdt-dev.txt", train=train, shuffle_batches=False)
    test = morpho_dataset.MorphoDataset("czech-pdt-test.txt", train=train, shuffle_batches=False)

    analyzer_dictionary = MorphoAnalyzer("czech-pdt-analysis-dictionary.txt")
    analyzer_guesser = MorphoAnalyzer("czech-pdt-analysis-guesser.txt")

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, len(train.factors[train.FORMS].alphabet), len(train.factors[train.LEMMAS].alphabet),
                      train.factors[train.LEMMAS].alphabet_map["<bow>"], train.factors[train.LEMMAS].alphabet_map["<eow>"])

    # Train
    for i in range(args.epochs):
        network.train_epoch(train, args.batch_size)

        #network.evaluate("dev", dev, args.batch_size)
        accuracy = network.evaluate("dev", dev, args.batch_size)
        print("{:.2f}".format(100 * accuracy))
        
    # Predict test data
    with open("{}/lemmatizer_sota_test.txt".format(args.logdir), "w") as test_file:
        #forms = test.factors[test.FORMS].strings
        #lemmas = network.predict(test, args.batch_size)
        #for s in range(len(forms)):
            #for i in range(len(forms[s])):
                #print("{}\t{}\t_".format(forms[s][i], lemmas[s][i]), file=test_file)
            #print("", file=test_file)
        
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
                #tag = test.factors[test.TAGS].words[tags[s][i]]
                #tag = dev.factors[dev.TAGS].words[tags[s][i]]

                # print('candidates', form, lemma, tag)

                # Use analyzer (optional)
                if args.anal:
                    lemma = find_analysis(form, lemam)


                #print("{}\t_\t{}".format(form, tag), file=test_file)
                print("{}\t{}\t".format(form, lemma), file=test_file)
    
                    #print("{}\t_\t{}".format(forms[s][i], test.factors[test.TAGS].words[tags[s][i]]), file=test_file)
                print("", file=test_file)

        
