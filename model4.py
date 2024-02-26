import re
from tensorflow.compat.v1.keras.preprocessing.sequence import pad_sequences
from tensorflow.compat.v1.keras.utils import to_categorical

from tensorflow.compat.v1.keras.layers import Input, Embedding, LSTM, Dense, Dense, Attention, Concatenate
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.callbacks import EarlyStopping
import tensorflow.compat.v1 as tf

from sklearn.model_selection import train_test_split

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
       
def dataPreprocessing():
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

    encoderInput = []
    for line in cleanQuestions:
        lst = []
        for word in line.split():
            if word not in vocabulary:
                lst.append(vocabulary["<OUT>"])
            else:
                lst.append(vocabulary[word])
        encoderInput.append(lst)

    decoderInput = []
    for line in cleanAnswers:
        lst = []
        for word in line.split():
            if word not in vocabulary:
                lst.append(vocabulary["<OUT>"])
            else:
                lst.append(vocabulary[word])
        decoderInput.append(lst)


    encoderInput = pad_sequences(encoderInput, 13, padding="post", truncating="post")           #Transforms list into a 2D numpy array of shape (number of sequences, length of each seauence). Used in NLP to ensure all sequences in batch are of same length. This, uses 0 (by default) for <PAD>! That is why we explicitly changed <PAD> token's number (in vocabulary) to 0  !
    decoderInput = pad_sequences(decoderInput, 13, padding="post", truncating="post")


    decoderFinalOutput = []
    for i in decoderInput:
        decoderFinalOutput.append(i[1:])  #Leave first SOS token
    decoderFinalOutput = pad_sequences(decoderFinalOutput, 13, padding="post", truncating="post")
    decoderFinalOutput = to_categorical(decoderFinalOutput, len(vocabulary))

    return encoderInput, decoderInput, decoderFinalOutput, vocabulary, inverseVocabulary

encoderInput, decoderInput, decoderFinalOutput, vocabulary, inverseVocabulary = dataPreprocessing()

trainEncoder, testEncoder, trainDecoder, testDecoder = train_test_split(encoderInput, decoderInput, test_size=0.2)
trainFinalDecoder, testFinalDecoder = train_test_split(decoderFinalOutput, test_size=0.2)

def seq2seq():
    enc_input = Input(shape=(13,))
    dec_input = Input(shape=(13,))

    embed = Embedding(len(vocabulary)+1,
                    output_dim=50, 
                    input_length=13,
                    trainable=True)  #Reduces dimensionality. Turns all 10,000 words in vocabulary into a single semantic vector of 50 dimensions
    enc_embed = embed(enc_input)
    enc_lstm = LSTM(400, return_sequences=True, return_state=True)
    enc_output, h, c = enc_lstm(enc_embed)
    enc_states = [h,c]

    dec_embed = embed(dec_input)
    dec_lstm = LSTM(400,return_sequences=True, return_state=True)
    dec_output, _, _ = dec_lstm(dec_embed, initial_state=enc_states)

    attention = Attention(name="attentionLayer1")
    attention_out = attention([dec_output,enc_output])  #attention_outout = same output size as encoder (not decoder) ; 400 values

    decoder_with_attention = Concatenate(axis=-1)([dec_output,attention_out])  #800 values total

    dense = Dense(len(vocabulary), activation="softmax")
    dense_output = dense(decoder_with_attention)

    model = Model([enc_input,dec_input],dense_output)

    model.compile(loss="categorical_crossentropy", metrics=["acc"],optimizer="adam")
    return model

model = seq2seq()

checkPoint = tf.keras.callbacks.ModelCheckpoint(
                                                        "./seq2seqChatbot3.h5",
                                                        monitor = "val_loss",
                                                        save_best_only=True,
                                                        mode='min',
                                                        save_freq='epoch'
                                                )
    

model.fit([trainEncoder,trainDecoder], trainFinalDecoder, validation_data=([testEncoder, testDecoder], testFinalDecoder), callbacks=[checkPoint], epochs=35)