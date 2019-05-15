from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import numpy 
import nltk

stopWords = set(stopwords.words('english')) #used default nltk stopwords list in english

class LemmaTokenizer(object): #modified given recommended class for lemmatization on sci-kit learn to return tokens which are atleast of length 3, are not punctuation symbols and are not number#s
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self,doc):
        return [self.wnl.lemmatize(word,pos='v') for word in word_tokenize(doc) if((word not in stopWords) and (word not in string.punctuation) and (not word.isdigit()) and (len(word)>=3))]


trainDataFile = open("train1.dat","r") #Given training file was split itno train1.dat and test1.dat to cross validate the model

#extracts sentiment values for each movie review in training data set
sentimentTrain = []
for line in trainDataFile:
    sentimentTrain.append(line[:2])
    
trainDataFile.seek(0,0) #Goes back to the beginning of the file
docRepresent = TfidfVectorizer(strip_accents='ascii',tokenizer=LemmaTokenizer(),max_df=0.98,min_df=0.01) #vectorizer uses term frequencies and inverse document frequencies with lemmatization 
trainData = docRepresent.fit_transform(trainDataFile) #Generates term frequencies for training data
#print(docRepresent.get_feature_names())

testDataFile = open("test1.dat","r")
docRepresentwithVocab = TfidfVectorizer(strip_accents='ascii',tokenizer=LemmaTokenizer(),vocabulary=docRepresent.get_feature_names()) #Vectorizer using same vocabulary as train data set

validateData = docRepresentwithVocab.fit_transform(testDataFile) 
print(docRepresentwithVocab.get_feature_names())

#extracts sentiment data for validation
testDataFile.seek(0,0)
sentimentValidate = []
for line in testDataFile:
    sentimentValidate.append(line[:2])

#testing optimal values of k
for k in range[50,350]:
    correctPredictions = 0 

    for i in range(0,1000):
        cosine_values = cosine_similarity(validateData[i],trainData).flatten().tolist() #computes distance between chosen test data against the entire train data set
        rankclasses = numpy.argsort(cosine_values).flatten().tolist()[::-1][:k] #list of indexes of k highest distances
    
        sentiment = 0 

        #computing sentiment by summing weighted distances
        for index in rankclasses:
            if sentimentTrain[index] == '+1':
                sentiment += 1*cosine_values[index]
            else:
                sentiment -= 1*cosine_values[index]
        
        sentiment = sentiment/k


        sentiment = 1 if sentiment>0 else -1
    
        #print("Predicted: "+str(sentiment)+" Original: "+sentimentValidate[i])
        if (sentiment == 1 and sentimentValidate[i] == '+1') or (sentiment == -1 and sentimentValidate[i] == '-1'):
                correctPredictions += 1

    print("correctPredictions: "+str(correctPredictions/300)+" for k= "+str(k))

trainDataFile.close()
testDataFile.close()

