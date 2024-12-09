# Sentiment-Driven Topic Analysis in ISIS Tweets: Exploring Sentiment-Topic Pairs through Semi-Supervised and Unsupervised LDA
## Introduction
The rise of social media has provided a platform for free expression of views on both positive and negative communication. One of the more troubling aspects is how extremist groups, such as ISIS, leverage platforms like X (formerly known as Twitter) to spread their propaganda, recruit members and coordinate activities. 
## Dataset
We will analyse a dataset of ISIS Tweets from Kaggle that contains tweets related to ISIS activity and narratives from January 2015 to January 2016. The dataset offers a comprehensive collection of tweets that are crucial for understanding how ISIS-affiliated individuals and groups use social media to spread their ideology, recruit members and coordination of movements and activities. 
## Objectives
The main objective of this project is to perform sentiment analysis on the ISIS-related tweets dataset by classifying the tweets into three categories: positive, negative, and neutral before they undergo further analysis to determine if they fall into more specific topics, such as propaganda, hate, and planned attacks etc.
### Usefulness of Analysis
Analysing sentiment in ISIS-related tweets is useful for several reasons: 
1.	Early Detection of Extremist Messaging: Identifying emotionally charged tweets that signal propaganda, recruitment efforts, or attack coordination allow authorities to intervene early. 
2.	Improved Content Moderation: Social media platforms can benefit from sentiment analysis to develop more nuanced content moderation strategies that go beyond keyword filtering and focus on context and intent. 
3.	Strategic Countermeasures: Law enforcement, intelligence and counter-terrorism agencies can use the analysis to detect emerging trends in online radicalization to refine their strategies and take pre-emptive measures.
## Methodology
1.	Data Preprocessing: Data cleaning and normalisation, exploratory data analysis (EDA), feature extraction using word embedding  
2.	Sentiment Analysis: Label tweets using VADER, AFINN and BERTweet into based on sentiments (Positive, Negative, Neutral), then performing additional EDA on respective results 
3.	Topic Modelling: (1) Semi-supervised learning through manually labelling a portion of the data with pre-defined seed topics and words through TF-IDF in LDA. (2) Apply LDA to discover topics within each sentiment group.  
4.	Analysis: Identify accounts and tweets that express pro-ISIS sentiments based on sentiment topic pair.   
## Experiments
### Dataset Description 
The dataset consists of 17,410 entries with the following columns: 
1.	name: name associated with the Twitter account 
2.	username: twitter handle 
3.	description: bio of the Twitter account  
4.	location: location of the user  
5.	followers: number of followers the account has 
6.	number statuses: number of tweets or statuses the account has posted 
7.	time: date/timestamp when the tweet was posted 
8.	tweets: content of the tweets 
Exploratory Data Analysis (EDA) is performed on the original tweets to analyse and identify issues that need to be addressed during data pre-processing. First, we calculate the number of words in each tweet in the dataset to see if there are outliers in the distribution.
### Data Preprocessing Techniques 
Data Cleaning using Regex: 
1. Translation-related text: As some of the original tweets are in Arabic, the tweets that begin with “ENGLISH TRANSLATIONS” will have these words removed.
2. Non-alphabetic characters: Strip all non-alphabetic characters except the “#” symbol used for hashtags.
3. Retweet indicators, handles, and URLs: Remove "RT @", “http”, “https”, and “www”.
4. Unnecessary text: Remove “amp” which stands for ampersand, used as a string literal to represent “&” in the dataset.
Hashtag and words separation:
1. Separate hashtags from other words and capture the remaining words within the tweets. 
2. Data Normalisation through Stemming and Lemmatisation:  
a. Option 1: Remove stop words and stem words using Porter stemmer. 
b. Option 2: Remove stop words and lemmatize words.

We experimented with the two options. The results show that Option 2 is more suitable for the dataset, to preserve the meaning of words in the tweet contents – an important requisite for effective sentiment analysis.After pre-processing, another round of EDA is performed on the processed tweets. First, we calculate the number of words in each tweet in the dataset to see if there are outliers in the distribution.We observe that the processed tweets display a somewhat normalised distribution.Next, we construct a word cloud based on the processed dataset to confirm that unnecessary words or characters have been removed.
### Data Annotation and Labelling 
We experimented with three different methods of sentiment analysis, namely: VADER, AFINN and BERTweet (based on BERT). Since ground-truth sentiment labels isn’t used in the dataset, the models are evaluated intuitively through sentiment-labelled data and corresponding text.
#### Sentiment Analysis using VADER, AFINN & BERTweet
The results of the sentiment analysis by VADER, AFINN and BERTweet were as follows:  
The nuanced sentiment detection with a higher volume of neutral tweets and reduced positivity bias with a smaller proportion of positive tweets compared to the other 2 showed that BERTweet was more superior. This is in line with how BERTweet as a transformer-based model, which was pre-trained on Twitter data, leverages context more effectively than lexicon-based models like VADAR and AFINN.  
### Data Partition Strategy
There is no data partitioning of the dataset as the above-mentioned methods for sentiment analysis do not require training, validation and evaluation of models. Since the BERTweet is a pre-trained transformer, and we are not training and validating the model further in our analysis.
### Data Preprocessing for Topic Modelling
It was observed that words from foreign languages, like French, contributed noise to the topic modeling results. Additionally, commonly used informal English words in tweets, along with special characters and spelling errors, added further noise. The data was pre-processed to remove frequently occurring foreign words and to handle spelling errors.
Preprocessing involved the following steps:
1. Removal of top French foreign words.
2. Removal of commonly used words in informal English.
3. Correcting words with special characters and spelling errors.
### Unsupervised LDA using Gensim
#### Implementing unsupervised LDA using Gensim
The Gensim package is a Python library designed for unsupervised topic modeling and document similarity analysis, with efficient implementations for tasks like LDA. To perform unsupervised LDA for all three sentiment tweets (Positive, Negative and, neutral) in Gensim, the following steps were followed 
1. After preprocessing text data, a dictionary and bag-of-words corpus was created.
2. LDAModel was used to extract relevant topics from the corpus.
#### Visualizing topic modelling results through pyLDAvis
The pyLDAvis library is a Python tool for interactive visualization of topic modeling results, where each topic is displayed as a circle on a 2D plane, with size indicating topic prevalence and distance indicating topic similarity. 
### Semi-Supervised LDA using Gensim
Results of unsupervised topic modeling tend to be noisy and may miss relevant topics. Hence it was essential to perform semi-supervised LDA. Semi-supervised LDA is better than unsupervised LDA because it integrates labeled data to guide topic formation. It helps produce more relevant and interpretable topics aligned with specific domains or categories. We proceed with identifying seed topics and seed words across the three sentiments.
#### Identifying Seed topics across sentiments
We broadly had the following research aims to guide topic modeling for all three sentiments.
1. Early Detection of Extremist Messaging – Differentiating between impeding attacks and recruitment instructions.
2. Improved Content Moderation – Differentiating radicalisation from other legitimate religious texts.
3. Strategic Countermeasures – Identify terrorist attack targets, counter-recruitment, counter-radicalization, and labelling of at-risk persons based on tweet follows or RT.
The themes corresponding to these three aims were identified as follows. 
1. Spreading Propaganda [Propaganda]
2. Attracting youths through recruitment messages [Recruitment] 
3. Radical Change in perception towards an individual or community [Radicalisation].
The seed topics and seed words within the sentiment tweets corresponding to these three themes were identified for all three sentiments (Positive, Negative & Neutral). 
#### Performing Semi-Supervised LDA using Gensim
We can follow the below steps to perform semi-supervised LDA for all three sentiment-based tweet using Gensim:
1. Use the existing dictionary and corpus created for unsupervised LDA.
2. Use TF-IDF to come up with the seed words and merge them with the seed topics. Refer to Appendix 2. 
3. Incorporate identified seed topics and seed words in the creation of the LDA model.
4. LDA model was used to extract topics from the corpus.
5. Visualize results of semi-supervised LDA using pyLDAvis. Inspect the impact the seed words and seed topics had in extraction of topics. 
#### Topic Modeling using GPT 4.0
As a final step, we uploaded the three sentiment-labeled tweet datasets to GPT-4.0, prompting it to identify key topics, associated themes, and corresponding seed words for each of them. The goal was to assess whether the topics identified by unsupervised and semi-supervised LDA aligned with those generated by GPT-4.0.
# Results and Analyses
In the context of topic modeling, a lower perplexity score indicates a better fit to the data, meaning the topics generated by the model more accurately represent the text data. On the other hand, higher coherence scores suggest that the topics make more sense to human readers because the words within each topic are more closely related. 
PERPLEXITY SCORES
1. In the Unsupervised LDA model, negative perplexity scores for all sentiments indicate a relatively good fit, with the "Neutral" sentiment having a better score (-9.32), than the others.
2. In the Semi-Supervised LDA model, the perplexity scores are higher and positive for the "Positive" sentiment, but lower for the "Neutral" and "Negative" sentiments, which suggests that the model struggles more with fitting the data compared to the Unsupervised LDA model.
COHERENCE SCORES
1. For both models, the "Positive" sentiment shows the highest coherence scores, indicating that the positive topics are the most interpretable and cohesive. In the Unsupervised LDA model, the "Positive" coherence score is 0.60, higher than both "Neutral" and "Negative."
2. The Semi-Supervised LDA model has a slightly lower coherence score for "Positive" at 0.56 but performs similarly to the Unsupervised LDA model. The supervised learning model may improve structure in topic relevance, though perhaps at some cost to coherence.
Overall, the Unsupervised LDA model performs better in terms of perplexity, while both models show comparable coherence scores.
## Comparison of Topics across three methods 
#### Sentiment-Topic Analysis
1. When we compare the topics within each sentiment, we observe that positive sentiment topics tend to converge on religious expressions, with terms suggesting non-violent but supportive messaging that promotes group identity and recruitment. Even for seemingly negative words like martyrdom, the topic suggested by LLM resulted in pairing martyrdom with heroism. For positive sentiments and topic pair, we may conclude that the sentiment-topic pairs identified are largely for propaganda and recruitment purposes. 
2. For neutral sentiment topics, they tend to converge on practical updates, like statements and updates on group activities. These narratives are factual in nature but nonetheless still include religious and ideological elements, positioning the group’s activities within a broader geopolitical context. In a practical sense, these topics may not invoke strong emotions from the audience but may appeal to individuals who prefer factual message in support of ISIS. 
3. For negative topics, across the different methods used, the convergence is along the line of military conflict, civilian impact with emotionally charged language and word patterns. The topics revolve around radicalization and fostering hostility. The focus on casualties and opposition creates a victim-persecutor dynamic, potentially appealing to individuals sympathetic to militant causes.
#### Business use-case analysis
1. For positive sentiment, content moderators can identify subtle propaganda and monitor recruitment messaging by tracking religious and community terms tied to ideological expressions. Social media platforms might focus on clusters that frequently use these terms, flagging content that indirectly supports extremist ideologies through non-violent messaging.
2. For neutral sentiment, early detection of extremist messaging can help intelligence agencies track territorial control and group activities. Understanding these narratives helps in assessing which geographic areas or groups are of interest to extremists. Also, social media platforms can improve content moderation by distinguishing legitimate news from covert propaganda.
3. For negative sentiment, counter-terrorism agencies and social media platforms can use this analysis to monitor narratives involving military conflicts and anti-Western sentiments. It could assist to track radicalization hotspots and recruitment points. Humanitarian organizations can use the insights to assess areas with increased civilian impact and allocate resources effectively.
# Discussion and Gap Analysis
One significant limitation was the dataset’s lack of labeled sentiment categories, which made it difficult to evaluate sentiment analysis models effectively. Without “ground-truth” sentiment labels, we lacked a reliable metric to validate the sentiment predictions. Furthermore, the dataset's small size and its inherent bias (leaning heavily toward pro-ISIS sentiment) caused an imbalance, skewing the sentiment distribution and reducing the representativeness of the data.
# Future Work
In our analysis, we did not consider French words that remained in the corpus. These words were in the corpus as the French language used alphabets, and were excluded from the Arabic-English translation of the original tweets. What could be done in future studies is to properly identify the different languages and be deliberate about the translation. For this analysis, we only used ISIS supporter tweets. In future works, study can be done to compare pro-ISIS against anti-ISIS sentiment-topic pair to have a clear differentiation between the two groups of users. This will help content moderators to get to the correct accounts as the central nodes for information dissemination


