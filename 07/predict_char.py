from collections import Counter
import tensorflow as tf
import numpy as np

def one_hot(seq, depth):
    arr = np.zeros([len(seq),depth],dtype=np.int32)
    for i, elem in enumerate(seq):
        hot = np.zeros(depth, dtype=np.int32)
        hot[elem] = 1
        arr[i] = hot
    return arr
      
with open('/home/liefe/hw/03/uppercase.txt', 'rt', encoding='utf-8') as f:
    data = f.read()

    # Set hyperparams
    sequence_length = 25  # fixed here like char window    
    sequence_dim = 75  # chars in char_distrib  (hot)
    data_sz = len(data)
    
    batches = data_sz // sequence_length
    data = data[0:batches*sequence_length+1]  # make even size
    chars = list(set(data))  
    vocab_sz = len(chars)  # real distinct chars    
    counts = Counter(data)    
    num_units = 100
    
    #char_to_idx = {c:i for c, i in enumerate(chars)}
    #idx_to_char = {i:c for c, i in enumerate(chars)}
    print("n = {}, real vocab size = {}".format(data_sz, vocab_sz))
    print("seq length = {}, batch size = {}, training vocab size = {}".format(sequence_length, batches, sequence_dim))
    
    #print(chars)
    
    char_distrib =  sorted(counts.items(), key=lambda x: x[1], reverse=True)
    #print(char_distrib)
    char_to_idx = {'unk':0}
    idx_to_char = {0:'unk'}
    # Map only most freq chars to int and others above vocab size threshold to <unk> (below)
    for i, (ch, f) in enumerate(char_distrib, len(char_to_idx)):
        char_to_idx[ch] = i
        idx_to_char[i] = ch
        if len(char_to_idx) >= sequence_dim: break
    print(char_to_idx)
    print(idx_to_char)
     
    
    # Train model
    sess = tf.Session()
    #inputs = tf.placeholder(tf.int32, [None, sequence_length, n_features])
    #outputs = tf.placeholder(tf.int32, [None, sequence_length, n_features])
    inputs = tf.placeholder(tf.int32, [None, sequence_length])
    outputs = tf.placeholder(tf.int32, [None, sequence_length])
   
    input_embedded = tf.one_hot(inputs, n_features) 
    
    session.run(tf.global_variables_initializer)
    rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units)
    
    # need init state?
    init_state = rnn_cell.zero_state(batch_size=sequence_length, dtype=tf.float32)    
    hidden_layer, last_states = tf.nn.dynamic_rnn(rnn_cell, inputs_embedded, initial_state=init_state, dtype=tf.float64)
    output_layer = tf.layers.dense(hidden_layer, sequence_dim)
    #tf.contrib.learn
    loss = tf.losses.sigmoid_cross_entropy(tf.cast(outputs,tf.int32), output_layer)
    loss = tf.losses.sparse_softmax_cross_entropy(outputs, output_layer)
    global_step = tf.train.create_global_step()
    optimizer = tf.train.AdamOptimizer().minimize(loss)                       
    epochs = 1 
    #epoch = 0
    i = 0 # char window
    sequence_n = 0  # training example/sequence   
    #sequences = []
    #labels = []
    
    for epoch in range(epochs):
        
        # Get next batch (all data sequences)
        sequences = np.zeros([batches, sequence_length, sequence_dim], np.int32)
        labels = np.zeros([batches, sequence_length, sequence_dim], np.int32)                
        for batch in range(batches):
            # New batch
            print('batch: ', batch)
            #if (i + sequence_length + 1) > data_sz:
                #epoch += 1
                #sequence_n = 0
                #print("first epoch finished")
                #break
                # reset RNN  
            
            #for t in range(sequence_length):
            sequence = [char_to_idx[ch] if ch in char_to_idx else 0 for ch in data[i:i+sequence_length]]
            x_hot = one_hot(sequence, depth=sequence_dim)
            label = [char_to_idx[ch] if ch in char_to_idx else 0 for ch in data[i+1:i+sequence_length+1]]                
            y_hot = one_hot(label, depth=sequence_dim)
            #print("seq", sequence)
            #print("label", label)            
            #print([idx_to_char[j] for j in sequence])
            #print([idx_to_char[j] for j in label])
            #print(x_hot)
            #print(y_hot)                
            sequences[batch, :, :] = x_hot             
            labels[batch, :, :] = y_hot
      
            #print(sequences[batch,:,:])
            #print(labels[batch,:,:])
            #break 
        
            # Run
            sess.run({inputs: sequences, outputs: labels})
            
                     
            #n += 1
            i += sequence_length   # advance window
        
        #print(sequences[-1,:,:])
        
            #sequences = np.zeros([batches, sequence_length, sequence_dim], np.int32)
            #labels = np.zeros([batches, sequence_length, sequence_dim], np.int32)

            #sequences.append(sequence)
            #labels.append(label)
         
            #sequence = [ch if ch in char_to_idx else 'unk' for ch in data[i:i+sequence_length]]
            #labels = [ch if ch in char_to_idx else 'unk' for ch in data[i+1:i+sequence_length+1]]
            
            
           
            # if n > 15: break
            
          
            #inputs = []
            #labels = []
        
        #for j in range(sequence_length):
            #ch = data[i+j]
            #if ch in char_to_idx:
                #x = char_to_idx[ch]
            #else:
                #x = char_to_idx['unk']
            #next_ch = data[i+j+1]
            #if next_ch in char_to_idx:
                #y = char_to_idx[next_ch]
            #else:
                #y = char_to_idx['unk']
                
    #print(sequences)
    #print(labels)
    #print(len(sequences))
    
    