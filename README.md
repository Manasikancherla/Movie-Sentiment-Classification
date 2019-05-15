# Movie-Sentiment-Classification

Infer sentiment (or polarity) from free-form review text submitted for movies

****************************************************

Approach:

Before pre-processing the given data, the text in the train data and test data files were
converted to lower-case. To verify the model that was being built, the train data file was
split into two .dat files: train1.dat (24000 reviews) and test1.dat (1000 reviews). For
preprocessing the data and calculating the distance between two records, the sklearn and
ntlk libraries were used. Each movie review was represented as a vector, and the collection
of reviews was represented as a sparse matrix. Term frequencies and inverse document
frequencies were used to represent each movie review. The dimensionality was reduced by
using lemmatization, and by removing stop words, punctuation symbols, one-lettered and
two-lettered words. Cosine similarity was used to compute the similarity between two
movie reviews. A weighted knn approach was used, where the class of the neighbor, as well 
as the distance between the test record and the neighbor were taken into consideration.

***************************************************

Methodology of choosing approach and parameters:

Initially, a very basic knn model was implemented where each review was represented
using word counts. Also, the data was not preprocessed before computing distances
between two reviews. The distance between two neighbors were not considered when
predicting the sentiments of the test data i.e. an unweighted knn classifier was used. These
problems were addressed in a sequential manner. However, even after resolving these
issues, the dimensionality of the data was very large. Then, lemmatization was used to
group similar terms together. Terms which occured in more than 97% of the reviews and
lesser than 2% of the reviews were ignored(arbitrarily assigned values), along with stop
words. This improved the accuracy of the model.

***************************************************

Computing Accuracy:

Accuracy is computed as the number of correctly predicted records divided by the total
number of records. It is a good metric for the movie sentiment classification problem
because both the classes hold equal importance. However, in the case that all the classes in
a classification problem don’t have the same importance, accuracy is not a useful metric.
For instance, if only one out of a thousand records belong to a class of highest importance,
even a 99% accurate model would be of no use.

********************************************************

Efficiency of algorithm:

Before using lemmatization and elimination of stop words, the program took over 3 hours
to predict sentiments for the test data. After including lemmatization, the preprocessing
time took upto 20 minutes, but the overall run time was roughly 2 hours.
