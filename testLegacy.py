import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Bidirectional, Dropout, TimeDistributed 
from tensorflow.compat.v1.keras.optimizers import Adam, SGD
from tensorflow.compat.v1.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import numpy as np


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
    text = re.sub(r"can't","can not", text)
    return text

def getVocabulary():
    

    lines = open("./dataset/movie_lines.txt", encoding='utf-8', errors='ignore').read().split("\n")
    convers = open("./dataset/movie_conversations.txt", encoding='utf-8', errors='ignore').read().split("\n")

    exchange = []
    for conver in convers:
        exchange.append(conver.split(" +++$+++ ")[-1][1:-1].replace("'","").replace(",","").split())

    dialog = {}
    for line in lines:
        dialog[line.split(" +++$+++ ")[0]] = line.split(" +++$+++ ")[-1]

    questions = []
    answers= []

    for conver in exchange:
        for i in range(len(conver) - 1):
            questions.append(dialog[conver[i]])
            answers.append(dialog[conver[i+1]])

    del(conver,convers,dialog,exchange,i,line,lines)

    sortedQuestions = []
    sortedAnswers = []

    for i in range(len(questions)):
        if len(questions[i]) < 13:
            sortedQuestions.append(questions[i])
            sortedAnswers.append(answers[i])


    cleanQuestions = []
    cleanAnswers= []

    for line in sortedQuestions:
        cleanQuestions.append(clean_text(line))

    for line in sortedAnswers:
        cleanAnswers.append(clean_text(line))

    for i in range(len(cleanAnswers)):
        cleanAnswers[i] = " ".join(cleanAnswers[i].split()[:11])

    del(answers,i,line,questions,sortedAnswers,sortedQuestions)

    cleanAnswers = cleanAnswers[:30000]   #From 31416 question and answers to only 30,000 question and answers
    cleanQuestions = cleanQuestions[:30000]


    word2count = {}
    for line in cleanQuestions:
        for word in line.split():
            if word not in word2count:
                word2count[word] = 1
            else:
                word2count[word] += 1
    for line in cleanAnswers:
        for word in line.split():
            if word not in word2count:
                word2count[word] = 1
            else:
                word2count[word] += 1

    threshold = 5
    vocabulary = {}
    wordNumber = 0
    for word, count in word2count.items():
        if count > threshold:
            vocabulary[word] = wordNumber
            wordNumber += 1


    for i in range(len(cleanAnswers)):
        cleanAnswers[i] = "<SOS> " + cleanAnswers[i] + " <EOS>"


    tokens = ["<PAD>","<EOS>","<OUT>","<SOS>"]
    x = len(vocabulary)
    for token in tokens:
        vocabulary[token] = x   #adding tokens to vocabulary
        x += 1

    vocabulary["cameron"] = vocabulary["<PAD>"]
    vocabulary["<PAD>"] = 0 

    inverseVocabulary = {w:v for v,w in vocabulary.items()}

    return vocabulary, inverseVocabulary

vocabulary, inverseVocabulary = getVocabulary()

def code(text):
        
        lst = []
        for word in text.split():
            if word not in vocabulary:
                lst.append(vocabulary["<OUT>"])
            else:
                lst.append(vocabulary[word])
        return lst

def decode(array):
    lst = []
    for num in array:
            if num not in inverseVocabulary:
                lst.append("<OUT>")
            else:
                lst.append(inverseVocabulary[num])
    return lst

seq2seq = tf.keras.models.load_model("./seq2seqChatbot.h5")
print(seq2seq.summary())



'''
    Why break down the encoder and decoder model, and not use seq2seq.predict()?
    To allow higher degree of control over token generation
    Allow beam search decoding
'''

'''
    They are 'sub models' derived from the main seq2seq model. They are built UPON them
    The inference models serve as an 'interface' to the main seq2seq model (like a portal to allow passing of inputs and extraction of outputs)
    They resides ON-TOP of the pretrained seq2seq, and main operations are performed by the master seq2seq model; using it's larned parameters!
    They only serve as interface to the seq2seq model. This interface is implemented only to allow higher manual control and power over generation of tokens 
    Huge power brought from this manual control
    
'''
def inference_encoder_model():
    '''
    Encoder persists in same architecture (along with trained parameters) underlyingly!
    We only 'elevating' input layer because we will pass the 'text' ourself.
    We elevate 3rd layer (LSTM). The only reason is to extract the hidden + cell state.
        - we dont connect the input and LSTM; THEY ARE ALREADY CONNECTED IN UNDERLYING ARCHITECTURE! we simply ELEVATE them to extract output
    Then, we define a interface-model using the elevated layers (needed to give elevated layers a proper shape) 
    
    '''
    encoder_inputs = seq2seq.input[0]  #First input = encoder input

    enc_output, enc_h, enc_c = seq2seq.layers[3].output    #the input is passed to the the LSTM layer. This lstm layer gives output, 'h' and 'c'. We will use 'h' and 'c'.
    encoder_states = [enc_h, enc_c]
    encoder_model = tf.keras.models.Model(inputs=encoder_inputs, outputs=encoder_states)
    return encoder_model

def inference_decoder_model():
    '''
        same explanation as for inference_encoder_model
        Main idea is 'elevation'.
        Main computation performed by the decoder INSIDE the seq2seq model; 
        this is only an interface to manually pass inputs and extract outputs (for higher degree of control)
    '''

    decoder_input = seq2seq.input[1]
    decoder_input_h = Input(shape=(400,), name="Input_Hidden")    #encoder LSTM Layer has 400 cells. There is a hidden and cell state for EACH cell. Therefore; 400 hidden and cell states!
    decoder_input_c = Input(shape=(400,), name="Input_Cell")
    decoder_states_input = [decoder_input_h,decoder_input_c]

    embed_layer = seq2seq.layers[2]  #number of embved layer is 2nd in model archetecture. Found using model.summary()
    decoder_embed = embed_layer(decoder_input)  #why explicitly embedding here? Search more. Something related to 'embedding each token generated'.

    decoder_lstm = seq2seq.layers[4]
    decoder_outputs, h, c = decoder_lstm(decoder_embed, initial_state=decoder_states_input)
    decoder_states = [h, c]

    dense = seq2seq.layers[5]

    decoder_outputs = dense(decoder_outputs)
    decoder_model = tf.keras.models.Model([decoder_input]+decoder_states_input, [decoder_outputs]+decoder_states)
    return decoder_model


encoderModel = inference_encoder_model()
decoderModel = inference_decoder_model()


def predict(input):
    input = clean_text(input)
    input = [code(input)]
    input = pad_sequences(input, 13, padding="post", truncating="post")


    state = encoderModel.predict(input)   #gives h and c states
    target_sequence = np.zeros((1,1))
    target_sequence[0,0] = vocabulary["<SOS>"]

    stop = False
    decoded_translation = ''



    while not stop:
        word_array, h, c = decoderModel.predict([target_sequence]+state)
        word_index = np.argmax(word_array[0,-1,:])
        word = inverseVocabulary[word_index]+ " "
        if word != "<EOS> ":
            decoded_translation += word
        if word == "<EOS> " or len(decoded_translation.split()) > 13:
            stop = True

        target_sequence = np.zeros((1,1))
        target_sequence[0,0] = word_index
        state = [h,c]

    return decoded_translation


while True:
    q = input()
    prediction = predict(q)
    print(prediction)
