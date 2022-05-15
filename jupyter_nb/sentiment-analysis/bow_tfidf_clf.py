# for data
import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt
import itertools

# for processing
import re
import nltk
from nltk.tokenize import word_tokenize
import string

# for bag-of-words
from sklearn import feature_extraction, feature_selection, model_selection, naive_bayes, pipeline, manifold, preprocessing, metrics, svm
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, RepeatedStratifiedKFold

# for explainer
from lime import lime_text

def get_wordnet_pos(treebank_tag):

	if treebank_tag.startswith('J'):
		return nltk.corpus.wordnet.ADJ
	elif treebank_tag.startswith('V'):
		return nltk.corpus.wordnet.VERB
	elif treebank_tag.startswith('N'):
		return nltk.corpus.wordnet.NOUN
	elif treebank_tag.startswith('R'):
		return nltk.corpus.wordnet.ADV
	else:
		return nltk.corpus.wordnet.NOUN

def my_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):

  # quitamos hashtags
  clean_tweet = re.sub('#[A-Za-z0-9_]+', '', text)

  # quitamos cashtags
  clean_tweet = re.sub('\$[A-Za-z0-9_]+', '', clean_tweet)

  # quitamos nosmbres de usuario
  clean_tweet = re.sub('@[A-Za-z0-9_]+', '', clean_tweet)

  # quitamos enlaces
  clean_tweet = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', clean_tweet)

  # quitamos signos de puntuacion
  clean_tweet = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', clean_tweet)

  # quitamos numeros
  clean_tweet = re.sub('[0-9_]+', '', clean_tweet)

  # quitar caracteres repetidos
  clean_tweet = re.sub(r'(.)\1{2,}', r'\1', clean_tweet)

  # tokenizamos
  clean_tweet = word_tokenize(clean_tweet)

  # filtramos palabras vacias
  if (lst_stopwords is not None):
    clean_tweet = [word for word in clean_tweet if word not in lst_stopwords]
  
  # Stemming 
  if (flg_stemm == True):
    ps = nltk.stem.porter.PorterStemmer()
    clean_tweet = [ps.stem(word) for word in clean_tweet]
                
  # Lematizacion
  if flg_lemm == True:
    wnl = nltk.stem.wordnet.WordNetLemmatizer()
    tags = nltk.pos_tag(clean_tweet)
    clean_tweet = [wnl.lemmatize(j[0],get_wordnet_pos(j[1])) for j in tags]
            
  # volvemos a jutar las palabras en una oracion
  text = " ".join(clean_tweet)
  return text

def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
  # clean (convert to lowercase and remove punctuations and characters and then strip)
  text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
  # Tokenize (convert from string to list)
  lst_text = text.split()    # remove Stopwords
  if lst_stopwords is not None:
    lst_text = [word for word in lst_text if word not in lst_stopwords]
                
  # Stemming (remove -ing, -ly, ...)
  if flg_stemm == True:
    ps = nltk.stem.porter.PorterStemmer()
    lst_text = [ps.stem(word) for word in lst_text]
                
  # Lemmatisation (convert the word into root word)
  if flg_lemm == True:
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    lst_text = [lem.lemmatize(word) for word in lst_text]
            
  # back to string from list
  text = " ".join(lst_text)
  return text

def grid_search_clf(df):
  #  https://stackoverflow.com/questions/40534082/sklearn-pipeline-fit-attributeerror-lower-not-found

  y = df['y'].values

  # Count (classic BoW)
  vectorizer_bow = feature_extraction.text.CountVectorizer(max_features=10000, ngram_range=(1,2))

  # Tf-Idf (advanced variant of BoW)
  vectorizer_tfidf = feature_extraction.text.TfidfVectorizer(max_features=10000, ngram_range=(1,2))

  """Ahora, vamos a procesar el corpus con alguna de las funciones anteriores para vectorizar nuestro corpus y analizar los vectores que se obtienen."""

  corpus = df["text_clean"]

  vectorizer_tfidf.fit(corpus) # realizamos la vectorizacion (bow con tfidf)
  X = vectorizer_tfidf.transform(corpus) #transformamos el conjunto de entrenamiento
  dic_vocabulary = vectorizer_tfidf.vocabulary_ # determinamos cual es nuestro vocabulario final

  # reduccion de dimensionalidad via prueba chi2

  X_names = vectorizer_tfidf.get_feature_names()
  p_value_limit = 0.95

  df_features = pd.DataFrame()

  for cat in np.unique(y):
    chi2, p = feature_selection.chi2(X, y==cat) 
    df_features = df_features.append(pd.DataFrame(
                     {"feature":X_names, "score":1-p, "y":cat})) 
    df_features = df_features.sort_values(["y","score"], 
                      ascending=[True,False])
    df_features = df_features[df_features["score"]>p_value_limit]
    
  X_names = df_features["feature"].unique().tolist()

  vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)
  vectorizer.fit(corpus)
  X = vectorizer.transform(corpus) 
  dic_vocabulary = vectorizer.vocabulary_

  # calculo de pesos para clases desbalanceadas
  c_w  = class_weight.compute_class_weight('balanced', classes = np.unique(y), y = y)
  c_w = dict(np.ndenumerate(c_w))

  # clasificador -> pueden cambiar el clasificador al que ustedes quieran, solo procuren cambiar los
  # parametros de la malla de busqueda
  clf = svm.SVC(class_weight='balanced', probability=False)

  # pipeline
  model = pipeline.Pipeline([("vectorizer", vectorizer),  ("classifier", clf)])

  # Set the grid parameters: ojo con la definicion ->  classifier__ // dos guiones bajos antes del parametro
  grid_params = {}
  grid_params['classifier__kernel'] = ['rbf']
  grid_params['classifier__gamma'] = np.arange(0.1,2.1,0.1)
  grid_params['classifier__C'] = np.arange(0.1,2.1,0.1)

  cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=1)
  grid_search = GridSearchCV(model, param_grid=grid_params, n_jobs=-1, cv=cv, scoring='balanced_accuracy')
  grid_result = grid_search.fit(df["text_clean"], y)
  print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
  means = grid_result.cv_results_['mean_test_score']
  stds = grid_result.cv_results_['std_test_score']
  params = grid_result.cv_results_['params']

  for mean, stdev, param in zip(means, stds, params):
      print("%f (%f) with: %r" % (mean, stdev, param))

def clf_data(df):

  # split train-test
  df_train, df_test = model_selection.train_test_split(df, test_size=0.25)

  # get target
  y = df['y'].values

  # Count (classic BoW)
  vectorizer_bow = feature_extraction.text.CountVectorizer(max_features=10000, ngram_range=(1,2))

  # Tf-Idf (advanced variant of BoW)
  vectorizer_tfidf = feature_extraction.text.TfidfVectorizer(max_features=10000, ngram_range=(1,2))

  """Ahora, vamos a procesar el corpus con alguna de las funciones anteriores para vectorizar nuestro corpus y analizar los vectores que se obtienen."""

  corpus = df["text_clean"]

  vectorizer_tfidf.fit(corpus) # realizamos la vectorizacion (bow con tfidf)
  X = vectorizer_tfidf.transform(corpus) #transformamos el conjunto de entrenamiento
  dic_vocabulary = vectorizer_tfidf.vocabulary_ # determinamos cual es nuestro vocabulario final

  # reduccion de dimensionalidad via prueba chi2

  X_names = vectorizer_tfidf.get_feature_names()
  p_value_limit = 0.95

  df_features = pd.DataFrame()

  for cat in np.unique(y):
    chi2, p = feature_selection.chi2(X, y==cat) 
    df_features = df_features.append(pd.DataFrame(
                     {"feature":X_names, "score":1-p, "y":cat})) 
    df_features = df_features.sort_values(["y","score"], 
                      ascending=[True,False])
    df_features = df_features[df_features["score"]>p_value_limit]
    
  X_names = df_features["feature"].unique().tolist()

  # Ya que reducimos el tamaño del vector que representa cada documento, necesitamos transformar cada vector de texto según la nueva representación. Noten que anteriormente definimos la misma función para vectorizar mediante TFIDF + BoW y BoW. La diferencia ahorita es que vamos a vectorizar con un vocabulario que le pasamos como argumento. Anteriormente usamos todas las palabras para aplicar reducción de dimensionalidad.

  vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)
  vectorizer.fit(corpus)
  X = vectorizer.transform(corpus) 
  dic_vocabulary = vectorizer.vocabulary_

  # calculo de pesos para clases desbalanceadas
  c_w  = class_weight.compute_class_weight('balanced', classes = np.unique(y), y= y)
  c_w = dict(np.ndenumerate(c_w))

  # comenzamos con el modelo de clasificacion

  # clf = naive_bayes.MultinomialNB()
  clf = svm.SVC(kernel = 'rbf', C = 1.8, gamma = 0.1, class_weight='balanced', probability=False)

  # pipeline
  model = pipeline.Pipeline([("vectorizer", vectorizer),  ("classifier", clf)])

  # entrenar y ajustar el clasificador
  predicted = cross_val_predict(model, df["text_clean"], y, cv=10)

  #model["classifier"].fit(X_train, y_train)

  clf_evaluation(y, predicted)

def plot2_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.1f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
	
	#plt.tight_layout()
	plt.ylabel('Clase correcta')
	plt.xlabel('Clase predicha')

def clf_evaluation(y_test, predicted):

  """Finalmente, vamos a evaluar el desempeño de nuestro clasificador. Calculamos las métricas básicas:

  * Accuracy
  * Precision
  * Recall
  * F1

  Además, determinamos nuestra matriz de confusión y la curva ROC.

  """
  classes = np.unique(y_test)

  # Accuracy, Precision, Recall
  accuracy = metrics.accuracy_score(y_test, predicted)
  print("Accuracy:",  round(accuracy,3))
  print("Detail:")
  print(metrics.classification_report(y_test, predicted))
    
  # Plot confusion matrix
  cfn = metrics.confusion_matrix(y_test, predicted)
  plt.figure()
  plot2_confusion_matrix(cfn, classes=classes, title=('Matriz de Confusión'))
  plt.figure()
  plot2_confusion_matrix(cfn, classes=classes, normalize=True, title=('Matriz de Confusión'))
  plt.show()

def explain_data(df_test, model, y_train, y_test, predicted):
  # Funcion para explicar las caracteristicas del texto
  # select observation

  i = 0
  txt_instance = df_test["text"].iloc[i]

  ## check true value and predicted value
  print("True:", y_test[i], "--> Pred:", predicted[i], "| Prob:", round(np.max(predicted_prob[i]),2))

  # show explanation
  explainer = lime_text.LimeTextExplainer(class_names=
               np.unique(y_train))
  explained = explainer.explain_instance(txt_instance, 
               model.predict_proba, num_features=3)
  explained.show_in_notebook(text=txt_instance, predict_proba=False)

def main():

  # Importamos los datos desde un archivo csv a un DataFrame de Pandas.
  df = pd.read_csv('csv_files/base_completa_v2_headers.csv')

  # Renombramos los encabezados para no escribir tanto en el futuro y seguir una nomenclatura aceptada dentro del área de ML.
  df = df.rename(columns={"tipologia":"y", "traduccion":"text"})

  # Mandamos a imprimir 5 filas de forma aleatoria para observar los datos.
  df.sample(5)

  # Aplicamos nuestra función para procesar el texto en el corpus dentro del DataFrame
  df["text_clean"] = df["text"].apply(lambda x: my_preprocess_text(x, flg_stemm=True, flg_lemm=False, 
                                                                      lst_stopwords=nltk.corpus.stopwords.words("english")))
  df.head()

  # Empezamos el proceso de clasificación
  clf_data(df)

if __name__ == '__main__':
  main()