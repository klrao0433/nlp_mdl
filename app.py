import torch.nn as nn
import pickle
import torch
import nltk
import numpy as np
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger') 
from flask import Flask, jsonify, make_response, request, render_template

app = Flask(__name__)

Vocab= pickle.load(open('charngram_saved_vocab_1109.pickle', 'rb'))

class classifier(nn.Module):
    
    #define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout):
        
        #Constructor
        super().__init__()    
              
        
        #embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        #lstm layer
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        
        #dense layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        
        #activation function
        self.act = nn.Sigmoid()
        
        
    def forward(self, text, text_lengths):
        
        #text = [batch size,sent_length]
        embedded = self.embedding(text)
        #embedded = [batch size, sent_len, emb dim]
                
        #packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths,batch_first=True)
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        #hidden = [batch size, num layers * num directions,hid dim]
        #cell = [batch size, num layers * num directions,hid dim]
        
        #concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
                
        #hidden = [batch size, hid dim * num directions]
        dense_outputs=self.fc(hidden)

        #Final activation function
        outputs=self.act(dense_outputs)
       # print(outputs.shape)
        
        return outputs

  #define hyperparameters
size_of_vocab = len(Vocab)
embedding_dim = 100
num_hidden_nodes = 32
num_output_nodes = 2
num_layers = 2
bidirection = True
dropout = 0.2

#instantiate the model
model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers, 
                   bidirectional = True, dropout = dropout)


#load weights
#path='/content/drive/My Drive/RNN/Text_classification_model weights/prepro2_charngram_saved_weights_new.pt'
model.load_state_dict(torch.load('prepro2_charngram_saved_weights_new.pt'));

#Model
#filename_1 = "/content/drive/My Drive/RNN/Text_classification_model weights/SVM_model.sav"
SVM_model = pickle.load(open('SVM_model.sav', 'rb'))

import spacy
nlp = spacy.load('en')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
from nltk.stem import WordNetLemmatizer
import numpy as np 
wordnet_lemmatizer = WordNetLemmatizer()
@app.route("/")
#def hello():
#    return "I am alive"

def home():
    return render_template('index.html')

def PredictionPreprocess(statement):
    tokenized = [tok.text for tok in nlp.tokenizer(statement)]
    lower = [t.lower() for t in tokenized]
    pun = [word for word in lower if word.isalnum()]
    lemm_ = [wordnet_lemmatizer.lemmatize(word,pos="v",) for word in pun]
    indexed_ = [Vocab[t] for t in lemm_] 
    return indexed_
  
def embeddingMatrix(indexedData):
    tensor_ = torch.LongTensor(indexedData)
    tensor_ = tensor_.unsqueeze(1).T 
    embedded = model.embedding(tensor_)
    return embedded

def MeanEmbedding(embedded_data):
    count=0
    for i in embedded_data:
      count=count+1
      c=torch.mean(i,axis = 0)
      if(count==1):
          tensornew = c.unsqueeze(0)
      else:
          tensornew = torch.cat((tensornew,c.unsqueeze(0)),dim=0)  
    pred_data= tensornew.detach().numpy()
    return pred_data

#models = []
output = []
names = []
def predict_ML(statement):
    indexed_matrix = PredictionPreprocess(statement)
    embedded_matrix = embeddingMatrix(indexed_matrix)
    mean_embedded= MeanEmbedding(embedded_matrix)
    pred_ = SVM_model.predict(mean_embedded)
    pred =  pred_[0]   
    #for i in pred_:
    #     a  =i
    #class_type_ = output
    #class_type = np.concatenate(class_type_, axis=0)
    #result_ = np.vstack(class_type_) 
    return pred
@app.route('/predict',methods=['POST'])
def predict():
    
    if request.method == 'POST':
        message = request.form.get('message')
        #data = message
    
    #input_text = request.form.get()
    
        my_prediction = int(predict_ML(message))
    #class__type = "Interface impact" if prediction == 0 else "no interface impact"

    #output = round(prediction[0], 2)

    return render_template('result.html', prediction = my_prediction)
    

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=4000)
