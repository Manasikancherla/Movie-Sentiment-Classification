from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import numpy
import nltk

stopWords = set(stopwords.words('english'))

class LemmaTokenizer(object): #class defined in sci-kit for lemmatization of training vocabulary
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self,doc):
        return [self.wnl.lemmatize(word,pos='v') for word in word_tokenize(doc) if((word not in stopWords) and (word not in string.punctuation) and (not word.isdigit()) and (len(word)>3))]


trainDataFile = open("train.dat","r")

sentimentTrain = []
for line in trainDataFile:
    sentimentTrain.append(line[:2])

trainDataFile.seek(0,0) #Goes back to the beginning of the file
docRepresent = TfidfVectorizer(strip_accents='ascii',tokenizer=LemmaTokenizer(),max_df=0.98,min_df=0.01) #vectorizer uses term frequencies and inverse document frequencies with lemmatization
trainData = docRepresent.fit_transform(trainDataFile) #Generates term frequencies for training data

testDataFile = open("test.dat","r")
docRepresentwithVocab = TfidfVectorizer(strip_accents='ascii',tokenizer=LemmaTokenizer(),vocabulary=docRepresent.get_feature_names()) #vectorizer uses the same vocabulary from train data to ext#ract features
testData = docRepresentwithVocab.fit_transform(testDataFile)

solution = open("solution.dat","w")
k = 180 #Set k value to desired value, used crossvalidify.py to select value of k

for i in range(0,25000):
    cosine_values = cosine_similarity(testData[i],trainData).flatten().tolist() #computes distance between chosen test data against the entire train data set
    rankclasses = numpy.argsort(cosine_values).flatten().tolist()[::-1][:k] #list of indexes of k highest distances

    sentiment = 0

    #computes sum of weighted distances multiplied by the value of the class
    for index in rankclasses:
        if sentimentTrain[index] == "+1":
            sentiment += 1*cosine_values[index]
        else:
            sentiment -= 1*cosine_values[index]

    sentiment = 1 if (sentiment/k)>0 else -1 #Assigns sentiment value depending on which value the weighted sum is closer to

    print("Sentiment value of test "+str(i)+": "+str(sentiment))   

    if sentiment == 1:
        solution.write("+1\n")
    else:
        solution.write("-1\n")

trainDataFile.close()
testDataFile.close()
solution.close()
