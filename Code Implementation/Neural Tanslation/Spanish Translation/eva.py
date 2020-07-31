from pickle import load
from numpy import array
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
import os
from pickle import dump
import unidecode
from unicodedata import normalize

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate target given source sequence
def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)[0]
	integers = [argmax(vector) for vector in prediction]
	target = list()
	for i in integers:
		word = word_for_id(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)

# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, test):
	actual, predicted = list(), list()
	for i, source in enumerate(sources):
		# translate encoded source text
		source = source.reshape((1, source.shape[0]))
		translation = predict_sequence(model, eng_tokenizer, source)
		print('Translated: [%s], Word=[%s]' % (translation, test[i]))


def load_doc(filename):
	with open(filename, 'r') as f:
		test = f.read().splitlines()
	return test

def clean_sentence(s):
	lst = s
	lst_cleaned = []
	for items in lst:
		items = normalize('NFD', items).encode('ascii', 'ignore')
		items = items.decode('UTF-8')
		#unaccented_string = unidecode.unidecode(items)
		lst_cleaned.append(items)
	return array(lst_cleaned)

def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# load datasets
dataset = load_clean_sentences('english-spanish-both.pkl')
train = load_clean_sentences('english-spanish-train.pkl')

#loading test dataset
filename = os.path.abspath('C:/Users/Sanju/Desktop/pics/translation/spanish/Datasets/test.txt')
test = load_doc(filename)
print(test)
cp = clean_sentence(test)
save_clean_data(cp,'test.pkl')
test = load_clean_sentences('test.pkl')
print(test)


# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
# prepare spanish tokenizer
spa_tokenizer = create_tokenizer(dataset[:, 1])
spa_vocab_size = len(spa_tokenizer.word_index) + 1
spa_length = max_length(dataset[:, 1])
# prepare data
trainX = encode_sequences(spa_tokenizer, spa_length, train[:, 1])
testX = encode_sequences(spa_tokenizer, spa_length, test)

# load model
model = load_model('model.h5')
# test on the test sequences
print('test')
evaluate_model(model, eng_tokenizer, testX,test)