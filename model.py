import numpy as np
import tensorflow as tf
import time
import re

def clean_text(text = ""):
    text = text.lower()
    text = re.sub(r"i'm","i am", text)
    text = re.sub(r"he's","he is", text)
    text = re.sub(r"she's","she is", text)
    text = re.sub(r"that's","that is", text)
    text = re.sub(r"what's","what is", text)
    text = re.sub(r"where's","where is", text)
    text = re.sub(r"\'ll"," will", text)
    text = re.sub(r"\'ve"," have", text)
    text = re.sub(r"\'re"," are", text)
    text = re.sub(r"\'d"," would", text)
    text = re.sub(r"won't","will not", text)
    text = re.sub(r"can't","cant", text)
    text = re.sub(r"wasn't","was not", text)
    text = re.sub(r"hasn't","has not", text)
    text = re.sub(r"it's","it is", text)
    text = re.sub(r"can't","cant", text)
    
    
    
    

    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,`'!]","", text)

    return text

def dataPreprocessing():
    lines = open("./dataset/movie_lines.txt", encoding='utf-8', errors='ignore').read().split("\n")
    conversation = open("./dataset/movie_conversations.txt", encoding='utf-8', errors='ignore').read().split("\n")
    
    #Dictionary mapping each line with id
    id2line = {}
    for line in lines:
        split_data = line.split("+++$+++")
        if len(split_data) == 5:  #Just making sure we take only those lines which indeed have 5 elements (line id, text, person etc). Just in case
            id, text = split_data[0].strip(), split_data[4].strip()
            id2line[id] = text
    
    #List of all conversations (id's)
    conversationCombination = []
    for entry in conversation:
        _entry = entry.split("+++$+++")
        if len(_entry) == 4:
            comb = list(_entry[3].strip())
            comb = comb[2:-2]
            comb = [x for x in comb if (x != "'") and (x != " ")]
            comb = "".join(comb).split(",")
            conversationCombination.append(comb)

    #Seperating Question (inputs) & Answer (target) ID's
    q = []
    a = []
    for entry in conversationCombination:
        for i in range(len(entry)-1):
            q.append(id2line[entry[i]])
            a.append(id2line[entry[i+1]])
    
    #Cleaning:
    clean_q = []
    clean_a = []
    for question in q:
        clean_q.append(clean_text(question))
    for answer in a:
        clean_a.append(clean_text(answer))
    

    #Making dictionary that maps each word to its occurance (Makes a dictionary of unique words, their count is used to filter low occuring words out):
    word2count = {}
    for q in clean_q:
        for word in q.split():
            if word not in word2count:
                word2count[word] = 1
            else:
                word2count[word] += 1
    for a in clean_a:
        for word in a.split():
            if word not in word2count:
                word2count[word] = 1
            else:
                word2count[word] += 1

    
    #Tokenization & Filtering (2 Dictionaries; one that maps question-words and one that maps answer-words --> a unique integer) [Creating unique-word --> integer dictionary]
    threshold = 20 #Below this count, words will be removed
    qWords2Int = {}
    word_number = 0
    for _word,count in word2count.items():
        if count > threshold:
            qWords2Int[_word] = word_number
            word_number += 1


    aWords2Int = {}
    word_number = 0
    for _word,count in word2count.items():
        if count > threshold:
            aWords2Int[_word] = word_number
            word_number += 1


    #Adding SOS, EOS, PADDING & OUT Token [PADDING = when sentence is short & we add PADDING. OUT = filtered out words (below threshold in Word2Int)]
    tokens = ['<PAD>','<EOS>','<OUT>','<SOS>']
    for token in tokens:
        qWords2Int[token] = len(qWords2Int) + 1
        aWords2Int[token] = len(aWords2Int) + 1


    #Inverse Dictionaries (integer  -->  word)  [only for answers]
    aInt2Word = {word: count for count,word in aWords2Int.items()}

    #Adding <EOS> at end of each answer:
    for i in range(len(clean_a)):
        clean_a[i] += " <EOS>"

    #Turning entire question & answer sentences to numbers:
    qInt = []
    for question in clean_q:
        intSentence = []
        words = question.split()
        for _w in words:
            if _w in qWords2Int:
                intSentence.append(qWords2Int[_w])
            else:
                intSentence.append(qWords2Int["<OUT>"])
        qInt.append(intSentence)

    aInt = []
    for answer in clean_a:
        intSentence = []
        words = answer.split()
        for _w in words:
            if _w in aWords2Int:
                intSentence.append(aWords2Int[_w])
            else:
                intSentence.append(aWords2Int["<OUT>"])
        aInt.append(intSentence)
    
    
    # *** Sorting Question & Answers by length of Questions (Speeds up training, and decreases Loss):
    sorted_clean_question = []
    sorted_clean_answers = []

    for length in range(1,26):
        for index,question in enumerate(qInt):
            if len(question) == length:
                sorted_clean_question.append(question)
                sorted_clean_answers.append(aInt[index]) #Thats why using enumerate, index idhar kaam aya

    #Question sorted from lowest length (1) to higher lengths (upto 26).
    #When you're teaching a child English, you dont start with entire sentences, you start with small words!!!  :D

    return sorted_clean_question, sorted_clean_answers, qWords2Int, aWords2Int, sorted_clean_question, sorted_clean_answers

q, a, qWord2Int, aWords2Int, sorted_clean_question, sorted_clean_answer = dataPreprocessing()

def model_inputs():
    input = tf.placeholder(tf.int32, [None, None], name = "input") #1st Param = datatype, 2nd Param = Dimension of tensor, 3rd Param = simply the name of the tensor
    target = tf.placeholder(tf.int32, [None, None], name = "target") 
    learningRate = tf.placeholder(tf.float32, name = "learning_rate") #simply a value, therefore no end param which specifies dimension.
    keep_prob = tf.placeholder(tf.float32, name = "keep_prob") #drop-out rate 
    return input, target,learningRate,keep_prob

def preprocessTarget(targets, word2Int, batch_size): #~
    left_side = tf.fill([batch_size,1],word2Int["<SOS>"]) #A vector on batch_sizex1; all <SOS> tokens. This vector will be added before a matrix that contains the Target-answers (model needs to know Start of sentence, this is one approach)
    right_side = tf.strided_slice(targets,[0,0],[batch_size,-1],[1,1]) #Extracts a subset of a tensor (which will be our targets). from [0,0] (first element, first row first column) to [batch_size,-1] (all rows and all columns except the last one; which is the 'EOS' token). Third argument is 'slide': How many cells we want to slide when doing extraction.
    target = tf.concat([left_side, right_side], axis = 1) #axis = 1 : horizontally add the vector (containing <SOS>) to the matrix; so each row starts with SOS
    return target


#Encoder Archetecture:
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size) #Creates a single RNN. rnn_size is the length of the vector that this LSTM will output. [single ANN-neuron outputs a single value (they are very simple), but LSTM are complex neurons (that have hidden gates in themselves for storage of memory). Therefore, they output a vector!].
                                                  #This is the foundational way of creating a single LSTM cell. tf.keras.layers.LSTM() creates an entire layer (and is a 'keras' way). This is the tensorflow way.
    lstm_dropout_combined = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob) # This a wrapper/container. It basically creates a container with both the lstm and the drop out (the combination)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout_combined] * num_layers) #Stacks the lstm+dropout-container vertically. Data flows down to up. Each unit in this vertical stack is lstm+dropout
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,     
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs,
                                                       dtype = tf.float32)
    
    '''
     input = sewuence of 4 time stamps = [t1,t2,t3,t4]
     cell_fw (& cell_bw) are both vertically stacked
     both cell_fw and cell_bw flow data from bottom-most neuron to top-most neuron. Output of top-most neuron is the final output.
     cell-fw process data : [t1,t2,t3,t4]
     cell-bw processes data : [t4,t3,t2,t1]    *Reason is to find relationship between future and past: Word that follows heavily depends on the previos word (girl = her, boy = him). Combining forward-processed and backward-processed data allows more comprhensive data capturing

    each unit in cell has a rnn_size of (lets say) 2 [returns vector containing 2 dimensions]
    we only consider the fnal output from each layer (output from top-most neuron)

    output cell_fw = [fw_1, fw_2] # Past --> Future (encrypyted!)
    output cell_bw = [bw_1,bw_2]  # Future --> Past (encrypyted!)

    final output = combination of both:
    [[fw_1,bw_1],[fw_2,bw_2]]  #Both past and the future; encrypted in this matrix; a complex relationship established.

    ~cell_fw = process sentence in order (I am a boy)
    ~cell_bw = process sentence in backward (boy a am I)

    '''

    #-----
    '''
     _ = output. It is the output of the LSTM at each time step [t1,t2,t3,t4]
     At the first time step; the output from the first both cells = [fw_t1,bw_t4]; at 2nd output = [fw_t2,bw_t3]
     The output is the combined-vector of all time-steps: [ [fw2_t1, bw2_t4], [fw2_t2, bw2_t3], [fw2_t3, bw2_t2], [fw2_t4, bw2_t1] ]

     encoder_state = summary of what the model has learned overall.
                     simply an array of 2 elemnts: [fw_final_state, bw_final_state] (what the layers has learned overall) !
     
    '''

    #-----
    '''
     if we visualize an encoder; it is simply two stacked pillars (one for processing data forward and one for processing data backward). And each pillar is of size num_layers; and each layer = 1 LSTM + 1 Dropout combined. Each layer outputs a vector of size rnn_size)
    '''

    '''
     The encoder cell Looks very simple, having only 2 pillars of stacked layers
     BUT 
     The complexity lies in the birectional Nature
     The seq2seq model will try to capture information for both the past and the future, and will try to relate it. This provides rich context
    '''



    return encoder_state

#Decoder Archetecture:
def decoder_rnn(decoder_embedded_input, decoder_embdeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        cell = tf.contrib.MultiRNNCell([lstm_dropout]*num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases
                                                                      )
        training_predictions = decode_training_set(encoder_state, cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size)
        decoding_scope.reuse_variables()
        test_prediction = decode_test_set(encoder_state,cell,decoder_embdeddings_matrix,word2int["<SOS>"],word2int["<EOS>"], sequence_length - 1, num_words, decoding_scope, output_function, keep_prob, batch_size)
        return training_predictions, test_prediction

#Decode training Test
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    '''
    attention keys = keys to be compared with target state
    attention values = used to contruct context vectors
    attention score function = used to compute similarity between keys and target state
    attention contruct = used to build attention state
    
    '''
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train"    
                                                                             )
    '''
    This is an attentional decoder function for the training of the dynamic RNN-decoder
    '''
    decoder_output, decoder_final_state,decoder_final_context_state =  tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, training_decoder_function, decoder_embedded_input,sequence_length,scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output,keep_prob)
    return output_function(decoder_output_dropout)

#Decoding test/validation dataset:
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, total_unique_words ,sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
   
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(
                                                                              output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix, 
                                                                              sos_id, 
                                                                              eos_id, 
                                                                              maximum_length,
                                                                              total_unique_words,
                                                                              name = "attn_dec_inf"    
                                                                             )
    '''
    This is an attentional decoder function for the training of the dynamic RNN-decoder
    '''
    test_prediction, decoder_final_state,decoder_final_context_state =  tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, test_decoder_function, scope = decoding_scope)
    return test_prediction


def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs, answers_num_words+1,encoder_embedding_size, initializer = tf.random_uniform_initializer(0,1) )
    encoder_state = encoder_rnn(encoder_embedded_input,rnn_size,num_layers,keep_prob, sequence_length)
    preprocessed_targets = preprocessTarget(targets, qWord2Int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words+1,decoder_embedding_size],0,1))  #initializing decoder _ embeeded matrix
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix,preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state,questions_num_words,sequence_length, rnn_size, num_layers, qWord2Int,keep_prob,batch_size)
    return training_predictions, test_predictions



# ------------ Training -----------------------------------------

#Hyper Parameters
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

#Defining Session
tf.reset_default_graph()
session = tf.InteractiveSession()

#Loading Model Inputs
inputs, targets, lr, keep_prob = model_inputs()

#Setting sequence length
sequence_length = tf.placeholder_with_default(input=25, shape=None, name="sequence_length") #25 maximum length

#Getting the shape of input tensor
input_shape = tf.shape(inputs)

#Getting the training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs,[-1]),targets, keep_prob,batch_size,sequence_length,len(qWord2Int),len(aWords2Int),encoding_embedding_size,decoding_embedding_size,rnn_size, num_layers, qWord2Int)

#Setting up loss error, the optimizer and Gradient Clipping
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,targets, tf.ones([input_shape[0],sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [( tf.clip_by_value(grad_tensor, -5.0,5.0) , grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)



#Padding the sequence with the <PAD> token
#Question : []"Who", "are", "you", "<PAD>", "<PAD>", "<PAD>", "<PAD>"]
#Answer:  ["<SOS>", "I", "am", "a", "bot", ".", "<EOS>"","<PAD>"]

def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequences) for sequences in batch_of_sequences]) #Largest length of the dataset
    return [ sequence + [word2int["<PAD>"]]*(max_sequence_length - len(sequence)) for sequence in batch_of_sequences]  #Adding the PAD token (number representation) to sequence (cleverly, in a single line)


#Splitting Data into batches of questions and answers:
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        end_index = start_index + batch_index
        questions_in_batch = questions[start_index : end_index]
        answers_in_batch = answers[start_index : end_index]
        padded_questions_in_batch = apply_padding(questions_in_batch, qWord2Int)
        padded_answers_in_batch = apply_padding(answers_in_batch,aWords2Int)
        
        padded_questions_in_batch = np.array(padded_questions_in_batch)
        padded_answers_in_batch = np.array(padded_answers_in_batch)

        yield padded_answers_in_batch, padded_answers_in_batch

#Splitting questions and aswers into training and validation sets:
training_validation_split = int(len(sorted_clean_question)*0.15)
training_questions = sorted_clean_question[training_validation_split : ]
training_answers = sorted_clean_answer[training_validation_split : ]

validation_questions = sorted_clean_question[ : training_validation_split]
validation_answers = sorted_clean_answer[ : training_validation_split]


#Training
batch_index_check_training_loss = 100
batch_index_check_validation_loss = (len(training_questions) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1,epochs+1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate( split_into_batches(training_questions, training_answers) ):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                                targets: padded_answers_in_batch,
                                                                                                lr: learning_rate,
                                                                                                sequence_length : padded_answers_in_batch.shape[1],
                                                                                                keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print("Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds".format(epoch,epochs,batch_index,len(training_questions)//batch_size,total_training_loss_error / batch_index_check_training_loss, int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate( split_into_batches(validation_questions, validation_answers) ):
                _, batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch, #no optimizer, since optimizer used in training
                                                                          targets: padded_answers_in_batch,
                                                                          lr: learning_rate,
                                                                          sequence_length : padded_answers_in_batch.shape[1],
                                                                          keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions)/batch_size)
            print("Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds".format(average_validation_loss_error, batch_time))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print("I speak better now")
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better. I need to Practise ,more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My Apologies, I cannot speak any better")
        break

