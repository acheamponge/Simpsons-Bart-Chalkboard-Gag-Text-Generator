import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")


with open('./bart-chalkboard-data.txt', 'r', encoding='utf-8') as file:
  data = file.read()
  
  
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # pre-pad sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
    return in_text




st.title('El Barto AI: A Text Generator for The Simpsons Chalkboard Gag')

image = Image.open('./1.jpg')

st.image(image, use_column_width=True)
st.write('This project is a Text Generator of words to create a new Chalkboard gag for The Simpsons')

n = st.number_input('Type the number of words you want generate', min_value=1, step=1 )


s = st.text_input('Type a word or words you want to generate after')

r = st.button('Say hello')

if s and n and r:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([data])
    # determine the vocabulary size
    max_length = 14
    st.header((generate_seq(loaded_model, tokenizer, max_length-1, s, n)))

elif s and not n:
    st.write('Please input information')

else:
    st.write('Please input a word and a number')
    
    
