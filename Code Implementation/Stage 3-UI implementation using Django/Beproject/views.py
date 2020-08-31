from django.shortcuts import render

from django.shortcuts import redirect
import os
from django.core.files import File
from django.http import HttpResponse

#from .forms import UploadImageForm
from django.core.files.storage import FileSystemStorage

from .forms import ImageUploadForm
from django.conf import settings
from .opencv import opencv #import our opencv_dface.py file

#spanish/french translation libraries
from pickle import load
from numpy import array
from numpy import argmax
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
import os
from pickle import dump
import unidecode
from unicodedata import normalize
import re

import numpy as np
import cv2
import pytesseract
import imutils

import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" 







def spanish_language(request):
	if request.method == 'POST':
		form = ImageUploadForm(request.POST, request.FILES)
		if form.is_valid(): # check form content are valid
			post = form.save(commit=False)
			post.save()

			#pass the image filename to opencv_dface function
			imageURL = settings.MEDIA_URL + form.instance.document.name
			opencv(settings.MEDIA_ROOT_URL + imageURL)

			"""Extraction
			text = pytesseract.image_to_string(final_image, lang="eng")
			print(text)
			text2 = re.findall(r'[A-Z]+', text)

			print("The final text for translation:")
			module_dir = os.path.dirname(__file__)  
			file_path = os.path.join(module_dir, 'test.txt')

			with open('test.txt', 'w+') as f:

				for text2 in text2:
					print(text2, end = ' ')
					f.write(text2 + ' ')

				f.seek(0)
				data = f.read()
				f.closed"""

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
				file = open('D:/BE Main project/djangoprojects/Projects/output.txt', 'w')
				#C:/Users/DELL/Dev/BEdjango/src/app/
				actual, predicted = list(), list()
				for i, source in enumerate(sources):
					# translate encoded source text
					source = source.reshape((1, source.shape[0]))
					translation = predict_sequence(model, eng_tokenizer, source)
					print('Predicted: [%s], Word=[%s]' % (translation, test[i]))
					
					file.write(str( '%s \n') % (translation))
					#file.close()s  
				

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
			dataset = load_clean_sentences('D:/BE Main project/djangoprojects/Projects/Beproject/english-spanish-both.pkl')
			train = load_clean_sentences('D:/BE Main project/djangoprojects/Projects/Beproject/english-spanish-train.pkl')

			#loading test dataset
			filename = os.path.abspath('D:/BE Main project/djangoprojects/Projects/test.txt')
			test = load_doc(filename)
			print(test)
			cp = clean_sentence(test)
			save_clean_data(cp,'D:/BE Main project/djangoprojects/Projects/Beproject/test.pkl')
			test = load_clean_sentences('D:/BE Main project/djangoprojects/Projects/Beproject/test.pkl')
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
			model = load_model('D:/BE Main project/djangoprojects/Projects/Beproject/model.h5')
			# test on some test sequences
			print('test')
			evaluate_model(model, eng_tokenizer, testX,test)

			module_dir = os.path.dirname(__file__)  
			file_path = os.path.join(module_dir, 'test.txt')

			with open('test.txt', 'r') as f1:
				data = f1.read()
				f1.closed

			#pass the result and form to template
			return render(request, 'spanish_language.html',{'form': form, 'post' : post, 'data' : data, 'file' : open('D:/BE Main project/djangoprojects/Projects//output.txt', 'r').read()})
			#now create dface.html
		  

	else: #when it shows firstly, it will come to this else statement
		form = ImageUploadForm()
		return render(request, 'spanish_language.html', {'form': form}) # pass image form to template, 
		#press save button on the form, the post if statement is satisfied


#french language
def french_language(request):
	if request.method == 'POST':
		form = ImageUploadForm(request.POST, request.FILES)
		if form.is_valid(): # check form content are valid
			post = form.save(commit=False)
			post.save()

			#pass the image filename to opencv_dface function
			imageURL = settings.MEDIA_URL + form.instance.document.name
			opencv(settings.MEDIA_ROOT_URL + imageURL)

			"""Extraction
			text = pytesseract.image_to_string(final_image, lang="eng")
			print(text)
			text3 = re.findall(r'[A-Z]+', text)

			#write  text to a file and read from the file
			module_dir = os.path.dirname(__file__)  
			file_path = os.path.join(module_dir, 'test1.txt')   #full path to text.
			
			with open('test1.txt', 'w+') as f:
				for text3 in text3:
					print(text3, end = ' ')
					f.write(text3 + ' ')
				f.seek(0)
				data = f.read()
				f.closed"""

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

				file = open('D:/BE Main project/djangoprojects/Projects/output1.txt', 'w')
				actual, predicted = list(), list()
				for i, source in enumerate(sources):
					# translate encoded source text
					source = source.reshape((1, source.shape[0]))
					translation = predict_sequence(model, eng_tokenizer, source)
					print('Predicted: [%s], Word=[%s]' % (translation, test[i]))

					file.write(str( '%s \n') % (translation))


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
			dataset = load_clean_sentences('D:/BE Main project/djangoprojects/Projects/Beproject/english-french-both.pkl')
			train = load_clean_sentences('D:/BE Main project/djangoprojects/Projects/Beproject/english-french-train.pkl')

			#loading test dataset
			filename = os.path.abspath('D:/BE Main project/djangoprojects/Projects/test.txt')
			test = load_doc(filename)
			print(test)
			cp = clean_sentence(test)
			save_clean_data(cp,'D:/BE Main project/djangoprojects/Projects/Beproject/test1.pkl')
			test = load_clean_sentences('D:/BE Main project/djangoprojects/Projects/Beproject/test1.pkl')
			print(test)


			# prepare english tokenizer
			eng_tokenizer = create_tokenizer(dataset[:, 0])
			eng_vocab_size = len(eng_tokenizer.word_index) + 1
			eng_length = max_length(dataset[:, 0])
			# prepare french tokenizer
			fra_tokenizer = create_tokenizer(dataset[:, 1])
			fra_vocab_size = len(fra_tokenizer.word_index) + 1
			fra_length = max_length(dataset[:, 1])
			# prepare data
			trainX = encode_sequences(fra_tokenizer, fra_length, train[:, 1])
			testX = encode_sequences(fra_tokenizer, fra_length, test)

			# load model
			model = load_model('D:/BE Main project/djangoprojects/Projects/Beproject/FrenchModel.h5')
			# test on some test sequences
			print('test')
			evaluate_model(model, eng_tokenizer, testX,test)

			module_dir = os.path.dirname(__file__)  
			file_path = os.path.join(module_dir, 'test.txt')

			with open('test.txt', 'r') as f1:
				data = f1.read()
				f1.closed

			#pass the result and form to template
			return render(request, 'french_language.html',{'form': form, 'post' : post, 'data' : data, 'file' : open('D:/BE Main project/djangoprojects/Projects/output1.txt', 'r').read()})
			#now create dface.html
		  

	else: #when it shows firstly, it will come to this else statement
		form = ImageUploadForm()
		return render(request, 'french_language.html', {'form': form}) # pass image form to template, 
		#press save button on the form, the post if statement is satisfied

