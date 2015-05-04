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
import numpy

name_to_index = { "conservative":0, "unbiased":1, "liberal":2}
    
    
def article_text_generator(lines):
    for line in lines:
        category, url = line.split(",")
        yield name_to_index[category], get_article_text(url)
    
def get_article_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text
    

if __name__ == '__main__':
    
    corpus = numpy.load("corpus.npy")
    target = numpy.load("training_target.npy")
    
    # input corpus into vectorizer
    print "Vectorizing training data.."
    vectorizer = CountVectorizer(min_df=1)
    X_training = vectorizer.fit_transform(corpus)
    
    print "Applying term-frequency inverse-document-frequency transform to training data..."
    # apply term-frequency x inverse document frequency transformer
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_training)
    
    print "Training Multinomial Naive Bayes Classifier around data..."
    clf = MultinomialNB().fit(X_train_tfidf, target)
  
    new_url = ""
    while new_url is not "quit":
        new_url = raw_input("Enter URL to classify: " )
        new_text = get_article_text(new_url)
        X_new = vectorizer.transform([new_text])
        
        X_new_tfidf = tfidf_transformer.transform(X_new)
        
        predicted = clf.predict(X_new_tfidf)
        print "Predicted category: {}".format(str(predicted[0]))
        
    
    
    
   
    