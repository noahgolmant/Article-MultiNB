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

name_to_index = { "conservative":0.0 , "liberal":1.0 }
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

    while True:
        new_url = raw_input("Enter URL to classify: " )
        if new_url == "quit":
            break
        new_text = get_article_text(new_url)
        #X_new = vectorizer.transform([new_text])
        
        #X_new_tfidf = tfidf_transformer.transform(X_new)
        
        #predicted = clf.predict(X_new_tfidf)
        
        # retrain text_clf
        # text_score = distance to 1, which is the liberal side
        text_score = text_clf.predict_proba([new_text])[0].tolist()[1]
        print "Prediction score: {}".format(str(text_score))
        
        #update training data
        numpy.append(corpus, new_text)
        numpy.append(target, text_score)
        
        text_clf = text_clf.fit(corpus, target)

        
        
    # update the corpus and target storage files
    numpy.save("corpus", corpus)
    numpy.save("training_target", target)
    
    
   
    