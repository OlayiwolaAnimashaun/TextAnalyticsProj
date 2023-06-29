import re, pandas as pd, numpy as np, matplotlib.pyplot as plt
import wordcloud, nltk
from nltk.stem import PorterStemmer

from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score, precision_score, accuracy_score

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SequentialFeatureSelector

import warnings
warnings.filterwarnings('ignore')

########################## vectorisation functions #######################################
def build_count_matrix(corpus):
    ## tokenise the corpus first
    tokenised_corpus = [nltk.word_tokenize(doc) for doc in corpus]
    ## a list of dictionaries aka frequency distributions
    freq_dists = []
    for doc in tokenised_corpus:
        ## for each document, generate a dictionary 
        ## of tokens as keys and respective counts as values
        token_count = {}
        for token in doc:
            ## if we already encountered this token and thus it is
            ## already in the dictionary, we increase its count by 1
            if token in token_count.keys():
                token_count[token] += 1
            ## otherwise, it is the first time it occurs, so we assign 
            ## the value of one to its count
            else: 
                token_count[token] = 1
        ## create a pd series from the dictionary generated for each doc
        ## and append it to the list
        freq_dists.append(pd.Series(token_count))
    ## once we have series for each doc, we use the list to build the 
    ## data frame and then replace the nans with 0, and return it
    matrix = pd.DataFrame(freq_dists)
    matrix = matrix.fillna(0)
    return matrix

## function to compute the lengths of the docs
def compute_doc_lengths(count_matrix):
    return count_matrix.sum(axis=1)

## function to generate a matrix with normalised frequencies
def build_tf_matrix(corpus):
    count_matrix = build_count_matrix(corpus)
    doc_lengths = compute_doc_lengths(count_matrix)
    return count_matrix.divide(doc_lengths, axis=0)

## function to compute the idfs of each term in a matrix
def compute_term_idfs(count_matrix):
    nis = count_matrix[count_matrix>0].count(axis=0)
    return np.log2(len(count_matrix)/nis)

## function to generate a matrix with tfidfs scores
def build_tfidf_matrix(docs):
    count_matrix = build_count_matrix(docs)
    doc_lengths = compute_doc_lengths(count_matrix)
    tf_matrix = count_matrix.divide(doc_lengths, axis=0)
    idfs = compute_term_idfs(count_matrix)
    tfidf_matrix = tf_matrix.multiply(idfs, axis=1)
    return tfidf_matrix.fillna(0)

##################### trainining/testing validation ##########################
def crossvalidate_model(clf, X, y, print_=True):
    scoring = ['accuracy', 'precision_macro', 'recall_macro']
    scores = cross_validate(clf, X, y, scoring=scoring)
    perf = []    
    for key in scores.keys():
        perf.append(scores[key].mean())
    if(print_):
        print("Accuracy: %0.2f" %(perf[2])) 
        print("Precision macro: %0.2f" %(perf[3])) 
        print("Recall macro: %0.2f" %(perf[4]))
    return perf[2], perf[3], perf[4]

##################### text understanding functions ##########################
def print_n_mostFrequent(topic, text, n):
    tokens = nltk.word_tokenize(text)
    counter = Counter(tokens)
    n_freq_features = counter.most_common(n)
    print(str(n) + " most frequent tokens in " + topic + ": ", n_freq_features)
    for f in n_freq_features:
        print("\tFrequency of", '"'+ f[0] + '"', 'is', f[1]/len(tokens))

def print_common_tokens(texts):
    topics_tokens = [[np.array(tokens) for tokens in nltk.word_tokenize(text)] 
                     for text in texts]
    intersection = topics_tokens[0]
    for n in range(1, len(topics_tokens)):
        intersection = np.intersect1d(intersection, topics_tokens[n])
    print('============ Common Features ==========\n',
          len(intersection), '\n', intersection, '\n')

def generate_cloud(text):
    cloud = wordcloud.WordCloud(width=700, height=700, 
                                background_color='black',
                                min_font_size=10).generate(text)
    plt.figure(figsize=(7, 7), facecolor=None)
    plt.imshow(cloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    
## returns the normalised count of a POS for each tagged document
def normalise_POS_counts(tagged_docs, pos):
    counts=[]
    for d in tagged_docs:
        count = 0
        for pair in d:
            if pair[1] == pos:
                count += 1
        counts.append(count)
    lengths = [len(d) for d in tagged_docs]
    return [count/length for count, length in zip(counts, lengths)]

## plots the frequency of a POS across texts/docs in a list
def plot_POS_freq(docs, pos, categories):
    tagged_docs = [nltk.pos_tag(nltk.word_tokenize(doc)) for doc in docs]
    normalised_counts = normalise_POS_counts(tagged_docs, pos)
    plt.bar(np.arange(len(docs)), normalised_counts, align='center')
    plt.xticks(np.arange(len(docs)), categories, rotation=40)
    plt.xlabel('Category')
    plt.ylabel(pos + " frequency")
    plt.title('Frequency distribution of ' + pos)  
    

################################ improve text quality functions #########################
## function to deal with contractions
def resolve_contractions(doc, CONTR_DICT):
    for key, value in CONTR_DICT.items():
        doc = re.sub(key, value, doc)
    return doc

## function to deal with negation modifiers
def resolve_negations(text):
    tagged_text = nltk.pos_tag(nltk.word_tokenize(text))
    pairs = []
    i = 0
    while i < len(tagged_text) - 1:
        pair1 = tagged_text[i]
        pair2 = tagged_text[i+1]
        if(
              (re.match(r'RB', pair[1]) and re.match(r'[Nn][Oo][Tt]?', pair[0]) and 
                  (pair2[1].startswith('VB') or pair2[1].startswith('JJ'))
              )
               or (re.match(r'DT', pair[1]) and re.match(r'[Nn][Oo][Tt]?', pair[0]) and
                        pair2[1].startswith('NN'))
        ):
            pairs.append(pair1[0] + "_" + pair2[0])
            i = i + 2
        else:
            pairs.append(pair1[0])
            i = i + 1
    pairs.append(tagged_text[len(tagged_text)-1][0])
    new_text = ' '.join(pairs)
    return re.sub(r' (?=[!.,?:;])', '', new_text)

################################ cleaning/preprocessing functions #########################
## function to carry out initial cleaning
def clean_doc(doc):
    ## replace paranthetical notes with an empty string
    doc = re.sub(r'(\(.+?\))+', '', doc)
    ## replace references with an empty string
    doc = re.sub(r'(\[.+?\])+', '', doc)
    ## tabs, carriage returns, new lines,
    doc = re.sub(r'\s+', ' ', doc)
    ## multiple spaces replaced with a single space
    doc = re.sub(r'\s{2,}', " ", doc)
    ## replace space before punctuation sign
    doc = re.sub(r' (?=[!\.,?:;])', "", doc)
    return doc

## function to carry out concept typing, resolve synonyms and word variations
def improve_bow(doc, replc_dict):
    for key in replc_dict.keys():
        for item in replc_dict[key]:
            doc = re.sub(item, key, doc, flags=re.IGNORECASE)
    return doc

## function to remove tokens using POS  tags
def remove_terms_by_POS(doc, tags_to_remove):
    tagged_doc = nltk.pos_tag(nltk.word_tokenize(doc)) ## (sea, 'NN')
    new_doc = [pair[0] for pair in tagged_doc if pair[1] not in tags_to_remove]
    new_doc = ' '.join(new_doc)
    ## replace space before punctuation sign
    return re.sub(r' (?=[!\.,?:;])', "", new_doc)

## function to lower case at the beginning of the sentence only
def lower_at_begining(doc):
    sents = nltk.sent_tokenize(doc)
    ##tokenised_sents = [nltk.word_tokenize(token) in sent for sent in sents]##
    tokenised_sents = [re.sub(sent[0], sent[0].lower(), sent)
                       for sent in sents]
    return ' '.join(tokenised_sents)

## function tot remove stop words
def remove_sw(doc, sw):
    tokens = nltk.word_tokenize(doc)
    return re.sub(r' (?=[!\.,?:;])', "",
                  ' '.join([token for token in tokens if token not in sw]))

## function tot remove short tokens
def remove_by_token_len(doc, n):
    tokens = nltk.word_tokenize(doc)
    return re.sub(r' (?=[!\.,?:;])', "",
                  ' '.join([token for token in tokens if len(token) > n]))

## function to remove digits
def remove_d(doc):
    return re.sub(r'\d+', '', doc)

## function to carry out stemming
def stem_doc(doc, stemmer):
    tokens = nltk.word_tokenize(doc)
    return ' '.join([stemmer.stem(t) for t in tokens])


############### feature selection functions ###############################
"""univariate feature selection: it takes a matrix and a set of labels, as well as
the scheme used (chi square or anova)and the number of features to retain"""
def univariate_selection(X, y, scheme, print_=True):
    selector = SelectKBest(scheme).fit(X, y)
    X_reduced= selector.transform(X)
    scores = list(selector.scores_)
    columns =  list(X.columns)
    counter = Counter({key:val for key, val, in zip(columns, scores)})
    if print_:
        n= 0
        for feature in counter.most_common():
            n = n + 1
            print(feature)
            if n == len(X_reduced):
                break
    return X_reduced

"""metatransformer selection: it takes a classifier that
outputs coefficients or feature importance,a matrix and a set of labels"""
def meta_selection(clf, X, y, print_=True):
    fitted = clf.fit(X, y)
    model = SelectFromModel(fitted, prefit=True)
    X_reduced = model.transform(X)
    if print_:
        scores = []
        columns =  list(X.columns)
        if isinstance(clf, DecisionTreeClassifier) or isinstance(clf, ExtraTreeClassifier):
            scores = list(clf.feature_importances_)
        elif isinstance(clf, LogisticRegression) or isinstance(clf, SVC):
            scores = clf.coef_[0]
        counter = Counter({key:val for key, val, in zip(columns, scores)})
        print(counter.most_common(X_reduced.shape[1]))
    return X_reduced

"""rfe selection: it takes a classifier that outputs coefficients or feature importance,
a matrix and a set of labels, as well as the number of features to retain and the step;
it should return the new matrix containing """
def rfe_selection(clf, X, y, n, step, print_=True):
    rfe = RFE(estimator=clf, n_features_to_select=n, step=step )
    rfe.fit(X, y)
    X_reduced = rfe.transform(X)
    if print_:
        scores = rfe.ranking_
        columns =  list(X.columns)        
        counter = Counter({key:val for key, val, in zip(columns, scores)})
        print(counter.most_common(n))
    return X_reduced

"""sequential selection: it takes a classifier, a matrix and a set of labels,
as well as the number of features to retain; it should return the
new matrix containing only the retained features"""
def sequential_selection(clf, X, y, n, print_=True):
    sfs = SequentialFeatureSelector(estimator=clf,n_features_to_select=n)
    sfs.fit(X, y)
    X_reduced = sfs.transform(X)
    if print_:
        scores = sfs.get_support(indices=True)
        columns =  list(X.columns)        
        counter = Counter({key:val for key, val, in zip(columns, scores)})
        print(counter.most_common(n))
    return X_reduced


############################### clustering visualisation functions #############################
""" function to visualise dendrograms """
from scipy.cluster.hierarchy import dendrogram, linkage
def display_dendrogram(data, method='single'):
    Z = linkage(data, method)
    plt.figure(figsize=(10,7))
    plt.xlabel('Documents')
    plt.ylabel('Distance')
    dendrogram(Z, labels=data.index, leaf_rotation=90)

""" visualise similarities between documents """
def plot_similarities(similarities):
    plt.figure(figsize=(10,7))
    plt.plot(similarities)
    plt.xlabel('Cosine Similarities between Documents')
    plt.xticks(np.arange(len(similarities)))

""" visualise clusters """
def plot_clusters(data, labels, title):
    clusters = pd.DataFrame([labels], columns=data.index)
    plt.figure(figsize=(10,7))
    plt.plot(clusters.iloc[0], marker='o', linewidth=0, markersize=10)
    plt.title(title)
    plt.xlabel('Documents')
    plt.ylabel('Cluster Labels')
    plt.xticks(list(clusters.columns))
    plt.yticks(np.arange(len(set(labels))))
    
############################## hyperparameter tuning function #######################
""" allows to carry out either greedy or random search; default is greedy """
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
def search_optimal_params(clf, X, y, params,approach='grid'):
    if approach == 'random':
        rand_search = RandomizedSearchCV(clf, param_distributions=params, n_iter=100)
        rand_search.fit(X, y) 
        return rand_search.best_params_, rand_search.best_score_ 
    grid_search = GridSearchCV(clf, param_grid=params)
    grid_search.fit(X, y) 
    return grid_search.best_params_, grid_search.best_score_ 
