'''
Created on May 3, 2015

Multinomial Naive Bayes classification of internet news articles

@author: Noah
'''

#import numpy
from newspaper import Article
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from nltk.stem.snowball import SnowballStemmer
import numpy

name_to_index = { "conservative":0, "unbiased":1, "liberal":2}
stop = stopwords.words('english') 
stemmer = SnowballStemmer("english")
    
def article_text_generator(lines):
    for line in lines:
        category, url = line.split(",")
        yield name_to_index[category], get_article_text(url)
    
def get_article_text(url):
    article = Article(url)
    article.download()
    article.parse()
    
    # filter out stop words
    # and store the stemmers
    stop_word_filtered_text = [stemmer.stem(w) for w in article.text.split() if w.lower() not in stop]
    
    return ' '.join(stop_word_filtered_text)
    

if __name__ == '__main__':
    
    corpus = numpy.load("corpus.npy")
    target = numpy.load("training_target.npy")
    
    text_clf = Pipeline([('vect', CountVectorizer(min_df=1)), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
    text_clf = text_clf.fit(corpus, target)

    new_url = ""
    while new_url is not "quit":
        new_url = raw_input("Enter URL to classify: " )
        new_text = get_article_text(new_url)
        #X_new = vectorizer.transform([new_text])
        
        #X_new_tfidf = tfidf_transformer.transform(X_new)
        
        #predicted = clf.predict(X_new_tfidf)
        predicted = text_clf.predict([new_text])
        print "Predicted category: {}".format(str(predicted[0]))
        
    
    
    
   
    