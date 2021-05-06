import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from tensorflow import constant, float32
from tensorflow.saved_model import load

from gensim.models.phrases import Phraser
from gensim.models import KeyedVectors

from re import sub, findall

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import mysql.connector
from mysql.connector import errorcode

import boto3

# ---------------------------------------------

# Connect to S3 and download files in /tmp/ directory
s3_bucket_name = os.environ['bucket']
s3_client = boto3.client('s3')

# Get phrasers and word2vec embedding
vocabulary_folder = 'vocabulary/'

key_bigrams = vocabulary_folder + 'bigrams'
key_trigrams = vocabulary_folder + 'trigrams'
key_w2v = vocabulary_folder + 'HEPword2vec'

if not os.path.isdir('/tmp/' + vocabulary_folder):
	os.mkdir('/tmp/' + vocabulary_folder)

	s3_client.download_file(s3_bucket_name,key_bigrams,'/tmp/'+key_bigrams)
	s3_client.download_file(s3_bucket_name,key_trigrams,'/tmp/'+key_trigrams)
	s3_client.download_file(s3_bucket_name,key_w2v,'/tmp/'+key_w2v)

# Get the CNN components
model_path = ['cnn_classifier/','1/','variables/']
model_dir = '/tmp/'

model_folder = 'cnn_classifier/1/'
variable_folder = model_folder + 'variables/'

key_model = model_folder + 'saved_model.pb'
key_variables = variable_folder + 'variables.data-00000-of-00001'
key_variables_index = variable_folder + 'variables.index'

if not os.path.isdir('/tmp/' + model_folder):
	for directory in model_path:
		model_dir += directory
		os.mkdir(model_dir)

	s3_client.download_file(s3_bucket_name,key_model,'/tmp/'+key_model)
	s3_client.download_file(s3_bucket_name,key_variables,'/tmp/'+key_variables)
	s3_client.download_file(s3_bucket_name,key_variables_index,'/tmp/'+key_variables_index)

# Get the label for the classifier
labels_folder = 'labels/'

key_labels = labels_folder + 'labels.npy'

if not os.path.isdir('/tmp/' + labels_folder):
	os.mkdir('/tmp/' + labels_folder)
	s3_client.download_file(s3_bucket_name,key_labels,'/tmp/'+key_labels)

# ---------------------------------------------

# Load the phrasers objects
phraser1 = Phraser.load('/tmp/'+key_bigrams)
phraser2 = Phraser.load('/tmp/'+key_trigrams)

phraser = lambda item : list(phraser2[phraser1[item]])

# Load the pre-trained word2vec model
w2v = KeyedVectors.load('/tmp/'+key_w2v)

# Load the pre-trained model
model = load('/tmp/'+model_folder)
infer = model.signatures["serving_default"]

# Load the labels
labels = np.load('/tmp/'+key_labels,allow_pickle=True)

# ----------------------------------------------

# RDS configuration details
config = {}
config['user'] = os.environ['db_username']
config['password'] = os.environ['db_password']
config['database'] = os.environ['db_name']
config['host'] = os.environ['db_host']

# ----------------------------------------------

wnl = WordNetLemmatizer()
stp_en = stopwords.words('english')

# Clean the text and use lemmatizer
def clean_text_lemmatize(item,lemmatizer,stopwords):

	# remove latex equations
	item = sub('\$+.*?\$+','',item)

	# tokenize and remove punctuation
	item = findall('[a-zA-Z0-9]+',item)

	# lowecase everything
	item = [word.lower() for word in item]

	# remove english stopwords
	item = [word for word in item if word not in stopwords]

	# lemmatize the words
	item = [lemmatizer.lemmatize(word) for word in item]

	return item

# Maps the words into the text to the corresponding index in the word2vec model
def hashing_trick(text,w2v):
	return [w2v.vocab[word].index+1 for word in text if word in w2v.vocab]

# Pre-process titles and abstracts before applying the classifier
def prerpocessing_text(iterator,w2v,wnl,stp_en):

	corpus = []

	for title, abstract, link in iterator:

		# Clean the titles and abstracts fetched from the database
		cleanded_title = clean_text_lemmatize(title,wnl,stp_en)
		cleanded_abstract = clean_text_lemmatize(abstract,wnl,stp_en)

		# Find common n-grams and replace them
		cleanded_title = phraser(cleanded_title)
		cleanded_abstract = phraser(cleanded_abstract)

		# Hashed title and abstract
		hashed_text = hashing_trick(cleanded_title+cleanded_abstract,w2v)

		corpus.append((hashed_text,link))

	return corpus

# Predict the labels for the new papers
def classify_papers(corpus,labels,treshold=0.5,missing_class_name='Unknown topic'):

	classified_list = []

	for text,link in corpus:

		# Use the trained classifier to predict the labels
		inferred_tensor = infer(constant([text],dtype=float32))
		inferred_probabilities = inferred_tensor['dense'].numpy().flatten()

		# Find most likely class
		class_index = np.argmax(inferred_probabilities)

		# Assign label to the paper
		if inferred_probabilities[class_index] > treshold:
			classified_list.append((labels[class_index],link))
		else:
			classified_list.append((missing_class_name,link))

	return classified_list

# ----------------------------------------------

def handler(event, context):
	
	try:
		cnx = mysql.connector.connect(**config)
	except mysql.connector.Error as err:
		if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
			print("Something is wrong with the credentials")
		elif err.errno == errorcode.ER_BAD_DB_ERROR:
			print("Database does not exist")
		else:
			print(err)
	else:
		print('Succesfully connected to the database')
		
		cursor = cnx.cursor()

		sql_recent_papers = ("SELECT title, abstract, link FROM papers "
							 "WHERE date = (SELECT MAX(date) FROM papers)"
							)

		cursor.execute(sql_recent_papers)

		# Pre-process titles and abstracts and classify the papers
		corpus = prerpocessing_text(cursor.fetchall(),w2v,wnl,stp_en)
		classified_list = classify_papers(corpus,labels)

		print('New papers have been classified')

		# Add labels to the database
		sql_classify = "UPDATE papers SET label = %s WHERE link = %s"

		for update_values in classified_list:
			cursor.execute(sql_classify, update_values)

		cnx.commit()

		# Close the database now that we have updated it
		cnx.close()

		print('Connection to the database closed')
