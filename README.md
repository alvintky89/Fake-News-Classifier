# CAPSTONE - Fake News Classifier
Author: Tan Kai Yong Alvin (DSI-32)

## 1. Introduction / Problem Statement

Have you came across an online article and was unable to decipher if the piece of news that was read was real or fake? 

The dissemination of fake news poses a serious threat to cohesion and social well-being as it fosters  polarization and the distrust amongst people in the community. For example, during the recent Covid-19 pandaemic, spreading of falsehood on the efficacy of vaccination as well as its long term side effects had severely undermined country's effort to counter Covid-19 as it spread fear and uncertainty amongst people leading to people shying away vaccination.

In recent years, we have witnessed an emergence and rise in fake news. With the large volume of fake news being disseminated through social media, manual classification and identification of fake news has become increasingly challenging. Hence, the aim of this project is to develop a model that is capable of classifying fake news with its performance measured by the accuracy and recall scores.

## 2.  Data Cleaning and Exploratory Data Analysis

The dataset comprises of two csv files, namely 'True-CSV' which contains 21,417 rows of 'Real News' and 'Fake_csv' which contains 23,481 of 'Fake News'. Both datasets were obtained from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?search=source). 

The initial cleaning and pre-processing performed focuses on the dropping of nulls and duplicates, identification of outliers (based on number of words of articles), identification of words with obvious correlation with the respective class. Finally,the headline and the article text are merged to form the text feature. 

## 3 Modeling

Multinomial Naive Bayes were ran as the baseline model. Following which PyCaret an auto ML technique was applied to the dataset to determine the best model. The best models (top 3) derived from the PyCaret are Light Gradient Boosting Machine, Logistic Regression and SVM (with Linear Kernel) with Count Vectorizer. in addition to the traditional ML models, Long Short Term Memoiry (LSTM), a neural network model (which takes into consideration the sequence of text) and Global Vector for Word Representation (GloVe) word embedding which (factors in the context of the text) was applied as an alternate model. The summary of the modeling performance is shown below:

|Model|Transformer|Train Score (Accuracy)|Test Score (Accuracy)|Recall|
|:---------:|:--------------:|:----------:|:----------:|:----------:|
|  Multinomial Naive Bayes (Baseline)  |      Count Vectorizer     |   0.9505  | 0.9543  | -  |
|   Multinomial Naive Bayes (Hyperparameter Tuning) |      TF-IDF Vectorizer     |   0.9344  |   0.9359 | - |
|   Light Gradient Boosting Machine   |      Count  Vectorizer    |  0.9903  |   0.9931  |0.9935
|  Logistics Regression  |     Count Vectorizer   |   0.9847  |   0.9873  | 0.9889  |
|  SVM Linear Kernel |     Count Vectorizer   |   0.9838  |   0.9836  | 0.9851  |
|   LSTM (2 x LSTM layer, 1 word embedding layer: best accuracy)  |     GloVe Embedding    |   0.9960 |   0.9950  | - |

XXX

|        Model        |    Accuracy | 
|:-------------------:|:--------------:|
|         Light Boosting Gradient Machine        |  0.8460|
|         Logistic Regression         |  0.8150 |
|         LSTM (with GloVe Embedding         | 0.5700 |


## 4 Sentiment Analysis

The creators of fake news use various stylistic tricks to promote the success of their creations, with one of them being to excite the sentiments of the recipients. This has led to sentiment analysis, the part of text analytics in charge of determining the polarity and strength of sentiments expressed in a text, to be used in fake news detection approaches, either as a basis of the system or as a complementary element. 

In the project, sentiment analysis will be applied to evaluate sentiment of the fake news article. The outcome of the sentiment analysis can enable organisation / government bodies to prioritise their efforts to counter the fake news. Fake news which spreads negative sentiments  (i.e. anger, disgust and fear) should be given more attention.

The outcome of the sentiment analysis shwed that the top emotions in a fake news are 'anger' and 'fear'. Out of 24,000 fake news, 16,000 were evaluated to show negative emotions. Apart from politician names, the top topics generated from sentiment analysis are 'Law Enforcement', .Race Matters' and 'Affordable Health Care Act'. Since the dataset used is US –based, it can be concluded from the words that was extracted that law enforcement, racial and health care affordability are the key source of discontentment in USA.

## 5 Conclusion and Recommendation

**Conclusion**

* The best model is **Light Gradient Boosting Machine** model with Count Vectorizer which has an accuracy score of 0.9903 and 0.9931 for train and test respectively.

* For the purpose of deployment, the **Logistic Regression** model which have an accuracy score of 0.985 (train score) and 0.987 (test score) respectively is selected instead due to it having comparable result with the best model and is more explanable.

* Sentiment analysis showed that fake news portrays a predominantly negative sentiment (e.g. anger and fear).

**Recommendations**

* Since social media is the key source of fake news, model training should not be limited to real news article from the mainstream newspaper (e.g. Reuters, BBC, The Straits Times), our model should be trained with news article (both real and fake) posted on social media platforms (e.g. twitter, facebook and etc).

* Other than social media, fake news generated by Artificial Intelligence are also emerging threats. As such, model should also be trained on fake news dataset generated by AI. 

* Given the right labelled dataset, this can be further expanded into a multi-classification problem. A piece of fake news may have carying level of factual correctness ranging from total untruth to deceptive news to manipulated news with some truth.A multi-classification would minimize problem of text which are partially false and contain misleading information being classified as true.


