# Sentiment-Analysis-on-Reviews-using-Linear-SVC-and-TF-IDF
## Dataset link: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Jewelry_v1_00.tsv.gz

## Feature Extraction
### TF-IDF (TF=Term Frequency and IDF=Inverse Document Frequency): 
It is a statistical measure that is used to evaluate the importance of a word in a document. It is a method of extracting the features from the text data.

`TF-IDF = Term Frequency X Inverse Document Frequency`.  

`Term Frequency` = number of times a word appears in a document. 

`Inverse Document Frequency` = measure of how common a word is across all documents.

TF-IDF is implemented using sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

TF-IDF is often used as a weighting factor in information retrieval and text mining. The weight of a word increases proportionally to the number of times the word appears in the document, but is offset by the frequency of the word in the corpus, which helps to adjust for the fact that some words are more common than others. TF-IDF can be used to calculate the relevance of a document for a given query. The higher the TF-IDF score of a document for a given query, the more relevant the document is for that.

## Model and Files 
Linear Support Vector Classifier (SVC) is a supervised machine learning algorithm for classification problems. It can be used for both binary and multiclass classification. SVC works by mapping data points into a high-dimensional space and then finding a hyperplane that best separates the points.
- `model train and export example.ipynb` shows how to preprocess the data and how to create and export the model. It also contains the Performance Report of the model. Use this to train and generate the model.
- `model use example.py` shows how to use the exported model. Run this and give some input text to get the score.
- `review classifier model.joblib` is the actual exported model consisting of the vectorizer and model. This is the model generated after running the Jupyter Notebook.
