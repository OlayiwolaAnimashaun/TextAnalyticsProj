#!/usr/bin/env python
# coding: utf-8

# In[120]:


import re, pandas as pd, numpy as np, requests, bs4, matplotlib.pyplot as plt
import wordcloud, nltk
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score, precision_score, accuracy_score
import wordcloud
import text_mining_utils as tm
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering 
from sklearn.feature_selection import f_classif


nltk.download('stopwords')
# ### Using a scraper, source 90 texts/documents/articles: 30 for each category; describe the process employed and state the source websites/pages

# In[123]:


##function to retrieve text data
def retrieve_text_data(url, elems):
    
    ## page: gets url 
    page = requests.get(url)
    
    ## page_data: stores url 
    page_data = page.text
    
    ## soup: stores all relevant url data and strips away html tags
    ## data: array to stores data
    soup = bs4.BeautifulSoup(page_data, "html.parser").body
    data = []
    
    ## The for loop scans through all the relevant elements and adds them into the data array
    for e in elems: 
        data += soup.find_all(e)
        
    ## data: the get_text function is used to retrieve the text content of all the relevant elements in the data array
    data = [el.get_text() for el in data]
    
    ## The resulting strings are joined together and a string is returned
    return ''.join(data)




   ## This array holds 30 urls that are relevant to DC Comics
dc_urls = [
    "https://screenrant.com/best-dc-comic-books-2021-according-reddit/",
    "https://www.cbr.com/dc-comics-best-2020/",
    "https://screenrant.com/best-dc-miniseries-2019/",
    "https://www.comicbookherald.com/best-dc-comics-of-2019/",
    "https://www.comicbookherald.com/best-dc-comics-of-2018/",
    "https://screenrant.com/best-dc-comics-heroes-of-all-time-according-to-ranker/",
    "https://www.ranker.com/crowdranked-list/best-dc-comics-heroes",
    "https://www.shortlist.com/lists/best-dc-characters-402104",
    "https://www.cbr.com/comic-genres-matched-members-justice-league/",
    "https://www.gamesradar.com/best-dc-comics-stories/",
    "https://libguides.colum.edu/comicsgraphicnovels/Genre",
    "https://www.dccomics.com/characters",
    "https://www.insider.com/best-dc-comic-heroes-2019-1",
    "https://www.comicbookherald.com/best-dc-comics-of-2017/",
    "https://medium.com/@2ndHandCopy/the-best-comics-of-2016-part-2-5-3267af59e4d3",
    "https://rarecomics.wordpress.com/top-50-dc-comics-of-2015/",
    "http://www.multiversitycomics.com/news-columns/the-ten-best-dc-comic-books-right-now/",
    "https://www.one37pm.com/culture/movies-tv/best-dc-comics",
    "https://www.toynk.com/blogs/news/best-dc-comics",
    "https://www.complex.com/pop-culture/the-best-dc-comics-of-all-time/",
    "https://www.cbr.com/best-dc-comics-all-time/",
    "https://www.comicbookherald.com/the-best-dc-comics-of-2021/",
    "https://www.comicbookherald.com/the-best-100-dc-comics-since-crisis-on-infinite-earths-1985/",
    "https://www.cheatsheet.com/entertainment/dc-comics-greatest-superheroes-of-all-time.html/",
    "https://wegotthiscovered.com/comicbooks/the-best-dc-comics-heroes/",
    "https://www.comicsbookcase.com/updates/best-comics-2022-dc-comics",
    "https://culturefly.com/blogs/culture-blog/best-dc-comic-storylines",
    "https://couchguysports.com/ranking-the-best-dc-comic-characters/",
    "https://www.jelly.deals/best-dc-comics-new-readers-superman-batman",
    "https://whatculture.com/comics/10-greatest-dc-superheroes-of-all-time",
    ]









    ## This array holds 30 urls that are relevant to Marvel Comics
marv_urls = [
    "https://www.comicsbookcase.com/updates/best-comics-2022-marvel",
    "https://www.toynk.com/blogs/news/best-marvel-comics",
    "https://www.comicbookherald.com/best-marvel-comics-of-2021/",
    "https://screenrant.com/best-marvel-comic-books-2021/",
    "https://www.cbr.com/marvel-comics-best-stories-releases-2020/"
    "https://www.comicbookherald.com/best-marvel-comics-of-2020/",
    "https://www.comicsbookcase.com/updates/best-comics-2020-marvel",
    "https://weirdsciencemarvelcomics.com/2021/01/03/marvel-comics-best-of-2020-year-in-review/",
    "https://www.comicbookherald.com/best-marvel-comics-of-2019/",
    "https://superherojunky.com/top-10-marvel-comics-of-2019/",
    "https://www.comicbookherald.com/best-marvel-comics-of-2018/",
    "https://www.comicbookherald.com/the-best-marvel-comics-of-2017/",
    "https://www.comicbookherald.com/the-best-marvel-comics-of-2016/",
    "https://www.pastemagazine.com/comics/the-10-best-comics-marvel-currently-publishes-2016/",
    "https://aminoapps.com/c/comics/page/blog/top-10-best-marvel-comics-of-2016/GMin_u0NWrEZb5mJ4LoZNwRD4EEYVe",
    "https://rarecomics.wordpress.com/top-50-marvel-comics-of-2015/",
    "https://www.gamesradar.com/best-marvel-comics-stories/",
    "https://www.wsj.com/articles/BL-SEB-85907",
    "https://www.gamesradar.com/marvel-characters/",
    "https://screenrant.com/best-marvel-comics-heroes-of-all-time-according-to-ranker/",
    "https://www.ranker.com/crowdranked-list/top-marvel-comics-superheroes",
    "https://www.toynk.com/blogs/news/best-marvel-characters",
    "https://lemonly.com/blog/top-10-most-popular-marvel-movie-characters",
    "https://fictionhorizon.com/20-best-marvel-characters-of-all-time/",
    "https://www.telltalesonline.com/28598/popular-marvel-characters/",
    "https://www.marvel.com/articles/culture-lifestyle/the-wider-world-of-marvel-genres",
    "https://screenrant.com/best-marvel-comic-books-ever-ranker/",
    "https://www.cbr.com/comic-genres-matched-members-avengers/",
    "https://www.marvel.com/comics/discover/1278/top-25-comics",
    "https://www.one37pm.com/culture/news/best-marvel-graphic-novels",
    "https://www.quora.com/Who-is-the-most-popular-Marvel-superhero",
    ]

    ## This array holds 30 urls that are relevant to the NBA
nba_urls = [
    "https://www.nbcsports.com/washington/wizards/2022-ranking-top-20-nba-players-right-now",
    "https://sportsnaut.com/best-nba-players-right-now/",
    "https://www.si.com/nba/2021/09/23/ranking-best-nba-players-top-100-2022-kevin-durant-giannis-antetokounmpo-lebron-james",
    "https://www.ranker.com/list/best-nba-players-2022/patrick-alexander",
    "https://www.washingtonpost.com/sports/interactive/2021/nba-top-100-players-2022/",
    "https://thegameday.com/41847/article/2021-nba-top-100-players-2022/",
    "https://www.si.com/nba/2020/12/14/top-100-nba-players-2021-daily-cover",
    "https://morningconsult.com/2021/10/18/nba-players-curry-durant-poll/",
    "https://www.theringer.com/nba/2021/5/4/22416337/top-25-nba-player-ranking-lebron-james-nikola-jokic",
    "https://www.complex.com/sports/best-nba-players-rankings",
    "https://www.persources.com/ranking-the-top-20-nba-players-2021/",
    "https://www.ranker.com/crowdranked-list/top-current-nba-players",
    "https://thesixersense.com/2021/09/17/nba-top-100-players-2021-22/",
    "https://www.statista.com/statistics/1266006/nba-top-shot-nft-most-popular-cards/",
    "https://www.interbasket.net/news/espns-100-best-nba-players-2020-21-nba-season-nbarank-list/31636/",
    "https://www.stadium-maps.com/facts/nba-teams-popularity.html"
    "https://www.lineups.com/articles/top-10-nba-players-in-the-2019-2020-season-kawhi-leonard-at-1/",
    "https://www.sportingnews.com/ca/nba/news/who-are-the-best-players-in-the-nba-entering-the-2020-21-season/4n84f58mc6sz157jdx4p2u712",
    "https://www.washingtonpost.com/graphics/2020/sports/nba-top-players-2020-2021/",
    "https://www.nbcsports.com/boston/celtics/nbas-top-100-players-2019-20-ranking-top-25",
    "https://www.si.com/nba/2018/09/10/top-100-nba-players-2019-lebron-james-stephen-curry-dirk-nowitzki",
    "https://www.sportingnews.com/in/nba/news/who-are-the-best-nba-players-entering-2019-20-season/13vm1p03wlnre14hecrk060vn9",
    "https://bleacherreport.com/articles/2889335-bleacher-reports-top-100-player-rankings-from-the-2019-20-nba-season",
    "https://www.insider.com/ranked-top-nba-players-right-now-2020-12",
    "https://www.si.com/extra-mustard/2022/02/17/lebron-james-lakers-lead-lids-jersey-sales",
    "https://www.sportskeeda.com/basketball/10-best-selling-nba-jerseys-2021-far",
    "https://www.nbcsports.com/washington/wizards/2022-nba-power-rankings-utah-jazz-take-top-spot-after-hot-streak",
    "https://wegrynenterprises.com/2021/10/12/report-ranking-the-most-popular-nba-teams/",
    "https://bolavip.com/en/nba/The-25-NBA-teams-with-most-fans-20200423-0002.html",
    "https://www.statista.com/statistics/240382/facebook-fans-of-national-basketball-association-teams/",
    "https://www.infoplease.com/us/basketball/top-grossing-nba-teams"
    ]


# In[113]:


retrieve_text_data("https://www.popcornbanter.com/5-best-dc-comics-stories-of-all-time/",
 ['h1', 'p'])


# ### Select the relevant html elements data, describing what you have retained, what you have removed and why; use the developer tools to aid your decisions

# In[122]:


## dc_docs: retrieves text data from headings labeled as heading 1 and paragraphs of urls
dc_docs = [retrieve_text_data(url, ['h1', 'p']) for url 
            in dc_urls]

print(len(dc_docs))
dc_docs


# In[124]:


## marv_docs: retrieves text data from headings labeled as heading 1 and paragraphs of urls
marv_docs = [retrieve_text_data(url, ['h1', 'p']) for url 
            in marv_urls]

print(len(marv_docs))
marv_docs


# In[116]:


c = 1
for url in dc_urls:
    print(retrieve_text_data(url, ['h1', 'p']))
    print('-------------------', c)
    c = c+1


# In[117]:


## nba_docs: retrieves text data from headings labeled as heading 1 and paragraphs of urls
nba_docs = [retrieve_text_data(url, ['h1', 'p']) for url 
            in nba_urls]

print(len(nba_docs))
nba_docs


# In[127]:


## all_docs: contains all previous url arrays
all_docs = dc_docs + marv_docs + nba_docs

## all_labels: contains a list of labels for all the previous url arrays
all_labels = (['DC'] * len(dc_docs) +
            ['Marvel'] * len(marv_docs) +
            ['NBA'] * len(nba_docs))
        
all_docs


# In[128]:


## prints all the labels 
len(all_docs), all_labels


# ### Build the corpus and explain the corresponding process

# In[129]:


def build_corpus(docs, labels):
    corpus = np.array(docs)
    corpus = pd.DataFrame({'Article': corpus, 'Class': labels})
    corpus = corpus.sample(len(corpus))
    return corpus

corpus = build_corpus(all_docs, all_labels)
corpus


# In[130]:


corpus.to_csv('corpus.csv', columns=['Article', 'Class'], index=False)


# ### Making the documents readable



# In[137]:


print(dc_text)


# In[138]:


print(marv_text)


# In[139]:


print(nba_text)


# ### Derive 3 matrices using 3 vectorisation techniques: counts, normalised counts and tfidf. Discuss the dimensionality and the differences between them

# In[3]:


corpus = pd.read_csv('corpus.csv')
corpus


# In[9]:


documents = list(corpus.Article)
baseline_count_matrix = tm.build_count_matrix(documents)
baseline_count_matrix


# In[11]:


baseline_tf_matrix = tm.build_tf_matrix(documents)
baseline_tf_matrix


# In[12]:


baseline_tfidf_matrix = tm.build_tfidf_matrix(documents)
baseline_tfidf_matrix


# ### Choose at least 1 classification algorithm for baseline modelling;

# In[15]:


dt_clf = DecisionTreeClassifier(random_state=1)
y = corpus.Class
y


# ### Apply the algorithm to the 3 matrices; document and discuss their performance using cross validation 

# In[17]:


tm.crossvalidate_model(dt_clf, baseline_count_matrix, y, print_=True)


# In[18]:


tm.crossvalidate_model(dt_clf, baseline_tf_matrix, y, print_=True)


# In[19]:


tm.crossvalidate_model(dt_clf, baseline_tfidf_matrix, y, print_=True)


# ### Derive word/token statistics for each category and explain what they indicate

# In[3]:


documents = list(corpus.Article)


# In[4]:


baseline_count_matrix = tm.build_count_matrix(documents)
baseline_count_matrix


# In[5]:


attributes = sorted(set(list(baseline_count_matrix.columns)))
print(attributes)


# In[6]:


dc_text = ' '.join(corpus.Article[corpus.Class == 'DC'])
marv_text = ' '.join(corpus.Article[corpus.Class == 'Marvel'])
nba_text = ' '.join(corpus.Article[corpus.Class == 'NBA'])


# In[7]:


tm.print_n_mostFrequent("DC", dc_text, 10)
tm.print_n_mostFrequent("MARVEL", marv_text, 10)
tm.print_n_mostFrequent("NBA", nba_text, 10)


# In[8]:


tm.print_common_tokens([dc_text, marv_text, nba_text]) 


# ### Use visualisations techniques (e.g., bar charts, word clouds) and identify frequently occuring terms, potential stop words, synonyms, concepts, and word variations comment on each topic/category

# In[9]:


texts = [dc_text, marv_text, nba_text]
tm.plot_POS_freq(texts, 'JJ', ['dc', 'marvel', 'nba'])


# In[10]:


tm.plot_POS_freq(texts, 'NN', ['dc', 'marvel', 'nba'])


# In[11]:


tm.plot_POS_freq(texts, 'DT', ['dc', 'marvel', 'nba'])


# In[12]:


tm.generate_cloud(dc_text)


# In[13]:


tm.generate_cloud(marv_text)


# In[14]:


tm.generate_cloud(nba_text)


# ###  Use 2 clustering algorithms with 2 different linkage schemes (e.g., minimum linkage vs. maximum linkage) and 2 different measures (e.g., symmetric vs. cosine) to identify the main clusters; give details of the algorithms, schemes and measures you tried, and what the results were: do they accurately identify the three clusters of text documents? If not, analyse the results to determine why not

# In[15]:


baseline_tfidf_matrix = tm.build_tfidf_matrix(documents)
baseline_tfidf_matrix


# In[16]:


y= corpus.Class
print(y)


# In[17]:


agg_single_cosine = AgglomerativeClustering(n_clusters=3, affinity='cosine',
                                     linkage='single')
agg_single_cosine.fit(baseline_tfidf_matrix)
agg_single_cosine_labels = agg_single_cosine.labels_
print(agg_single_cosine_labels)
print(list(y))


# In[18]:


agg_single_symmetric = AgglomerativeClustering(n_clusters=3, affinity='manhattan',
                                     linkage='single')
agg_single_symmetric.fit(baseline_tfidf_matrix)
agg_single_symmetric_labels = agg_single_symmetric.labels_
print(agg_single_symmetric_labels)
print(list(y))


# In[19]:


agg_complete_cosine = AgglomerativeClustering(n_clusters=3, affinity='cosine',
                                     linkage='complete')
agg_complete_cosine.fit(baseline_tfidf_matrix)
agg_complete_cosine_labels = agg_complete_cosine.labels_
print(agg_complete_cosine_labels)
print(list(y))


# In[20]:


agg_complete_symmetry = AgglomerativeClustering(n_clusters=3, affinity='manhattan',
                                     linkage='complete')
agg_complete_symmetry.fit(baseline_tfidf_matrix)
agg_complete_symmetry_labels = agg_complete_symmetry.labels_
print(agg_complete_symmetry_labels)
print(list(y))


# In[28]:


#do k++ clustering
km_plus = KMeans(n_clusters=3, random_state=1, )
km_plus.fit(baseline_tfidf_matrix)
km_plus.fit_predict(baseline_tfidf_matrix)
#obtain the labels
plus_cluster_labels = km_plus.labels_
##compare the cluster labels with the actual labels
print(plus_cluster_labels)
print(list(y))


# In[ ]:


#do k++ clustering
km_plus = KMeans(n_clusters=3, random_state=1, )
km_plus.fit(baseline_tfidf_matrix)
km_plus.fit_predict(baseline_tfidf_matrix)
#obtain the labels
plus_cluster_labels = km_plus.labels_
##compare the cluster labels with the actual labels
print(plus_cluster_labels)
print(list(y))


# ### Text preprocessing tasks: what preprocessing tasks are the most suitable for your data? Choose at least 3 tasks based on your findings from data understanding and discuss why they might be suitable. Document and discuss the incremental performance after each applied technique to the 3 matrices and decide whether they should be included in the final pipeline (justify your decisions) 

# In[4]:


#Initial cleaning of data
clean_data = corpus.copy()
clean_data.Article = clean_data.Article.apply(tm.clean_doc)
clean_data


# In[5]:


clean_count_matrix = tm.build_count_matrix(list(clean_data.Article))
tm.crossvalidate_model(dt_clf, clean_count_matrix, y, print_=True)
print("No. of terms after cleaning:", clean_count_matrix.shape[1])


# In[6]:


clean_tf_matrix = tm.build_tf_matrix(list(clean_data.Article))
tm.crossvalidate_model(dt_clf, clean_tf_matrix, y, print_=True)
print("No. of terms after cleaning:", clean_tf_matrix.shape[1])


# In[7]:


clean_tfidf_matrix = tm.build_tfidf_matrix(list(clean_data.Article))
tm.crossvalidate_model(dt_clf, clean_tfidf_matrix, y, print_=True)
print("No. of terms after cleaning:", clean_tfidf_matrix.shape[1])


# In[8]:


# Stop words removal
improved_data = clean_data.copy()
universal_sw = nltk.corpus.stopwords.words('english')
print(universal_sw)


# In[9]:


swr_u_data = improved_data.copy()
swr_u_data.Article = swr_u_data.Article.apply(tm.remove_sw, sw=universal_sw)

swr_u_count_matrix = tm.build_count_matrix(list(swr_u_data.Article))
tm.crossvalidate_model(dt_clf, swr_u_count_matrix, y, print_=True)
print("No. of terms after removal:", swr_u_count_matrix.shape[1])


# In[10]:


swr_u_tf_matrix = tm.build_tf_matrix(list(swr_u_data.Article))
tm.crossvalidate_model(dt_clf, swr_u_tf_matrix, y, print_=True)
print("No. of terms after removal:", swr_u_tf_matrix.shape[1])


# In[11]:


swr_u_tfidf_matrix = tm.build_tfidf_matrix(list(swr_u_data.Article))
tm.crossvalidate_model(dt_clf, swr_u_tfidf_matrix, y, print_=True)
print("No. of terms after removal:", swr_u_tfidf_matrix.shape[1])


# In[12]:


custom_sw = ['the', 'of', 'and', 'to', 'in', 'is', 'was', 'on', 's']
swr_c_data = improved_data.copy()

swr_c_data.Article = swr_c_data.Article.apply(tm.remove_sw, sw=custom_sw)
swr_c_count_matrix = tm.build_count_matrix(list(swr_c_data.Article))

tm.crossvalidate_model(dt_clf, swr_c_count_matrix, y, print_=True)
print("No. of terms after removal:", swr_c_count_matrix.shape[1])


# In[13]:


swr_c_tf_matrix = tm.build_tf_matrix(list(swr_c_data.Article))
tm.crossvalidate_model(dt_clf, swr_c_tf_matrix, y, print_=True)
print("No. of terms after removal:", swr_c_tf_matrix.shape[1])


# In[14]:


swr_c_tfidf_matrix = tm.build_tfidf_matrix(list(swr_c_data.Article))
tm.crossvalidate_model(dt_clf, swr_c_tfidf_matrix, y, print_=True)
print("No. of terms after removal:", swr_c_tfidf_matrix.shape[1])


# In[15]:


#Improving the BOW
repl_dictionary = {
    'comics': ['comic(s)[-]books', 'stories'],
    'superhero':['superheroes', 'hero(es)'],
    'writer': ['author(s)', 'creator(s)'],
    'NBA': ['league'],
    'team': ['franchise(s)'],
    'season': ['year']
}

improved_data.Article = improved_data.Article.apply(tm.improve_bow, replc_dict=repl_dictionary)

improved_count_matrix = tm.build_count_matrix(list(improved_data.Article))

tm.crossvalidate_model(dt_clf, improved_count_matrix, y, print_=True)
print("No. of terms after improving the bow:", improved_count_matrix.shape[1])


# In[16]:


improved_tf_matrix = tm.build_tf_matrix(list(improved_data.Article))
tm.crossvalidate_model(dt_clf, improved_tf_matrix, y, print_=True)
print("No. of terms after improving the bow:", improved_tf_matrix.shape[1])


# In[17]:


improved_tfidf_matrix = tm.build_tfidf_matrix(list(improved_data.Article))
tm.crossvalidate_model(dt_clf, improved_tfidf_matrix, y, print_=True)
print("No. of terms after improving the bow:", improved_tfidf_matrix.shape[1])


# ### Algorithms-based Feature selection/reduction tasks: Choose at least 2 techniques to try. Document  and discuss the performance after each applied technique and decidewhich one to include in the final p ipeline (justify your decision); of the terms chosen by the algorithms as being the most predictive, do they concur with the terms you thought would be the  best predictors from data understanding?

# In[18]:


## Univariate Feature Selection
uni_data = improved_data.copy()
uni_tfidf_matrix = tm.build_tfidf_matrix(
    list(uni_data.Article))
uni_reduced_tfidf_matrix = tm.univariate_selection(
    uni_tfidf_matrix, uni_data.Class, scheme=f_classif)
uni_reduced_tfidf_scores = tm.crossvalidate_model(
    dt_clf, uni_reduced_tfidf_matrix, y)
print("No. of terms after applying anova feature selection:", 
      uni_reduced_tfidf_matrix.shape[0])


# In[19]:


# RFE
rfe_data = improved_data.copy()
rfe_tfidf_matrix = tm.build_tfidf_matrix(
    list(rfe_data.Article))
rfe_reduced_tfidf_matrix = tm.rfe_selection(
    dt_clf, rfe_tfidf_matrix, y, n=100, step=2)
rfe_tfidf_scores = tm.crossvalidate_model(
    dt_clf, rfe_reduced_tfidf_matrix, y)
print("No. of terms after rfe:", 
      rfe_reduced_tfidf_matrix.shape[1])


# In[4]:


## Hyperparameter Tuning
params = {
    "criterion": ['gini', 'entropy'],
    "max_depth": range(3, 16),
    "min_samples_split": range(2, 16),
    "min_samples_leaf": range(3, 10),
    "min_impurity_decrease": [0.01, 0.02, 0.03, 0.04, 0.05]
}


# In[5]:


## Baseline Count Matrix
documents = list(corpus.Article)
baseline_count_matrix = tm.build_count_matrix(documents)

baseline_count_scores = tm.crossvalidate_model(dt_clf, baseline_count_matrix, y, print_=True)


# In[7]:


## change the params of the DT to the optimal ones above
opt_baseline_count_clf = DecisionTreeClassifier(random_state=1,
                                      criterion='gini',
                                      max_depth=3,
                                      min_impurity_decrease=0.01,
                                      min_samples_split=2,
                                      min_samples_leaf=5)

## retrain and get performance
opt_baseline_count_scores = tm.crossvalidate_model(opt_baseline_count_clf,
                                         baseline_count_matrix,
                                         y)


# In[8]:


## Baseline TF Matrix
baseline_tf_matrix = tm.build_tf_matrix(documents)
tm.crossvalidate_model(dt_clf, baseline_tf_matrix, y, print_=True)


# In[12]:


## change the params of the DT to the optimal ones above
opt_baseline_tf_clf = DecisionTreeClassifier(random_state=1,
                                      criterion='gini',
                                      max_depth=4,
                                      min_impurity_decrease=0.01,
                                      min_samples_split=2,
                                      min_samples_leaf=2)

## retrain and get performance
opt_baseline_tf_scores = tm.crossvalidate_model(opt_baseline_tf_clf,
                                         baseline_count_matrix,
                                         y)


# In[ ]:


## Baseline TFIDF Matrix
baseline_tfidf_matrix = tm.build_tfidf_matrix(documents)
tm.crossvalidate_model(dt_clf, baseline_tfidf_matrix, y, print_=True)


# In[ ]:





# In[4]:


#Initial cleaning of data
clean_data = corpus.copy()
clean_data.Article = clean_data.Article.apply(tm.clean_doc)
clean_data


# In[8]:


clean_count_matrix = tm.build_count_matrix(list(clean_data.Article))
tm.crossvalidate_model(dt_clf, clean_count_matrix, y, print_=True)
print("No. of terms after cleaning:", clean_count_matrix.shape[1])


# In[ ]:


tm.search_optimal_params(dt_clf, clean_count_matrix,
                        y, params)


# In[5]:


improved_data = clean_data.copy()


# In[14]:


# RFE
rfe_data = improved_data.copy()
rfe_tfidf_matrix = tm.build_tfidf_matrix(
    list(rfe_data.Article))
rfe_reduced_tfidf_matrix = tm.rfe_selection(
    dt_clf, rfe_tfidf_matrix, y, n=100, step=2)
rfe_tfidf_scores = tm.crossvalidate_model(
    dt_clf, rfe_reduced_tfidf_matrix, y)
print("No. of terms after rfe:", 
      rfe_reduced_tfidf_matrix.shape[1])


# In[15]:




tm.search_optimal_params(dt_clf, rfe_reduced_tfidf_matrix,
                        y, params)


# In[17]:


## change the params of the DT to the optimal ones above
opt_tfidf_clf = DecisionTreeClassifier(random_state=1,
                                      criterion='gini',
                                      max_depth=3,
                                      min_impurity_decrease=0.01,
                                      min_samples_split=2,
                                      min_samples_leaf=5)

## retrain and get performance
opt_tfidf_scores = tm.crossvalidate_model(opt_tfidf_clf,
                                         rfe_reduced_tfidf_matrix,
                                         y)


# In[11]:


## Univariate Feature Selection
uni_data = improved_data.copy()
uni_tfidf_matrix = tm.build_tfidf_matrix(
    list(uni_data.Article))
uni_reduced_tfidf_matrix = tm.univariate_selection(
    uni_tfidf_matrix, uni_data.Class, scheme=f_classif)
uni_reduced_tfidf_scores = tm.crossvalidate_model(
    dt_clf, uni_reduced_tfidf_matrix, y)
print("No. of terms after applying anova feature selection:", 
      uni_reduced_tfidf_matrix.shape[0])


# In[12]:


## Hyperparameter Tuning
params = {
    "criterion": ['gini', 'entropy'],
    "max_depth": range(3, 16),
    "min_samples_split": range(2, 16),
    "min_samples_leaf": range(3, 10),
    "min_impurity_decrease": [0.01, 0.02, 0.03, 0.04, 0.05]
}

tm.search_optimal_params(dt_clf, uni_reduced_tfidf_matrix,
                        y, params)


# In[13]:


## change the params of the DT to the optimal ones above
opt_tfidf_clf = DecisionTreeClassifier(random_state=1,
                                      criterion='gini',
                                      max_depth=3,
                                      min_impurity_decrease=0.01,
                                      min_samples_split=2,
                                      min_samples_leaf=3)

## retrain and get performance
opt_tfidf_scores = tm.crossvalidate_model(opt_tfidf_clf,
                                         uni_reduced_tfidf_matrix,
                                         y)


