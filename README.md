# Sentiment-Analysis-on-Reviews-using-Linear-SVC-and-TF-IDF
## Dataset link: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Jewelry_v1_00.tsv.gz

## Feature Extraction
TF-IDF: It is a method of extracting the features from the text data. TF stands for Term Frequency and IDF stands for Inverse Document Frequency.
TF-IDF is implemented using sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
TF-IDF (term frequency-inverse document frequency) is a statistical measure that is used to evaluate the importance of a word in a document. TF-IDF is a product of two measures: term frequency and inverse document frequency. 

Term frequency is the number of times a word appears in a document. 
Inverse document frequency is a measure of how common a word is across all documents.

TF-IDF is often used as a weighting factor in information retrieval and text mining. The weight of a word increases proportionally to the number of times the word appears in the document, but is offset by the frequency of the word in the corpus, which helps to adjust for the fact that some words are more common than others.

TF-IDF can be used to calculate the relevance of a document for a given query. The higher the TF-IDF score of a document for a given query, the more relevant the document is for that.

Linear Support Vector Classifier (SVC) is a supervised machine learning algorithm for classification problems. It can be used for both binary and multiclass classification. SVC works by mapping data points into a high-dimensional space and then finding a hyperplane that best separates the points.

1- model train and export example.ipynb shows how to preprocess the data and how to create and export the model. It also contains the Performance Report of the model.
2- model use example.py shows how to use the exported model.
3- review classifier model.joblib is the actual exported model consisting of the vectorizer and model.
