🧠 COVID-19 Tweet Sentiment Analysis & Topic Modeling
An end-to-end NLP project exploring public sentiment during the COVID-19 pandemic using Twitter data.
This repository combines data cleaning, sentiment relabeling, topic modeling with KMeans, and supervised classification (Naive Bayes, Logistic Regression, Linear SVC, and Ensemble Voting) to reveal key patterns in how people expressed emotions during the crisis.


🔽 1. Data Import & Preprocessing
📦 Description:
We load the original COVID-19 tweet dataset, perform initial cleaning (removal of URLs, emojis, and mentions), and prepare the text for NLP analysis.

🧰 Key steps:

Import dataset from CSV

Remove noise: links, emojis, mentions

Tokenization & lowercasing

Stopwords removal and lemmatization

📂 Output: Cleaned DataFrame ready for vectorization and modeling.


🧾 2. Sentiment Relabeling & Distribution Analysis
🎯 Description:
The original dataset includes nuanced sentiment labels such as Extremely Positive, Extremely Negative, and Neutral. To improve classification performance and reduce sparsity, we mapped them into three consolidated categories: Positive, Negative, and Neutral.

🔁 Relabeling Mapping:

Extremely Positive → Positive

Extremely Negative → Negative

Neutral → Neutral

📊 Before & After Comparison:

Visual	Description
fig-01-sentiment-distribution-before-after-mapping.png	Distribution of sentiment classes before and after mapping into three main categories

🧠 Insight:
After relabeling, the dataset showed a notable skew towards negative sentiment, reflecting public concern during the COVID-19 pandemic.


🕒 3. Temporal Analysis of Tweet Sentiment
📈 Description:
We analyzed how the distribution of tweet sentiments evolved over time during the pandemic. Using tweet timestamps, we plotted a stacked timeline of sentiment categories across the entire dataset.

📊 Visualization:

Visual	Description
fig-02-tweet-sentiment-timeline-stacked.png	Stacked timeline showing the evolution of Positive, Negative, and Neutral tweets during the COVID-19 period

🔍 Insight:
The timeline reveals sentiment spikes that align with major COVID-19 events (e.g. lockdowns, vaccine announcements), with negative sentiment dominating most periods.


🧩 4. Topic Modeling with KMeans Clustering
📌 Description:
To identify underlying themes in COVID-19 tweets, we applied TF-IDF vectorization followed by KMeans clustering. This unsupervised approach allowed us to group tweets based on similar content and discover latent topics in the data.

🔧 Methodology:

TF-IDF vectorization (n-grams: 1–2)

KMeans clustering with optimized k = 5

Top terms per cluster used for topic labeling

📊 Visualizations:

Visual	Description
fig-03-kmeans-topic-distribution.png	Number of tweets per cluster/topic
fig-04-sentiment-distribution-by-topic.png	Sentiment composition within each topic cluster

🧠 Insight:
Certain clusters (e.g. related to politics or restrictions) were dominated by negative sentiment, while others (e.g. recovery or support) showed more positive tones.


🤖 5. Sentiment Classification with Machine Learning Models
🧪 Description:
We trained several supervised classifiers to predict tweet sentiment using TF-IDF features. The task was framed as a 3-class classification problem: Positive, Negative, Neutral.

🛠️ Models Used:

Naive Bayes (BNB)

Logistic Regression

Linear SVC

Voting Ensembles (Hard & Soft)

⚙️ Evaluation Metric:
We used F1-score (macro) due to class imbalance and the importance of balanced precision/recall.


📉 6. Model Performance & Evaluation
📊 Confusion Matrices and F1 Scores:

Visual	Model	F1 Score
fig-06-logistic-regression-results-f1score-065.png	Logistic Regression	0.65
fig-07-linear-svc-results-f1score-079.png	Linear SVC	0.79
fig-08-hard-voting-ensemble-results-f1score-080.png	Hard Voting Ensemble	0.80
fig-09-soft-voting-ensemble-results-f1score-079.png	Soft Voting Ensemble	0.79
fig-10-naive-bayes-results-f1score-076.png	Naive Bayes (BNB)	0.76

🏆 Best Performing Model:
The Hard Voting Ensemble achieved the highest F1-score (0.80), outperforming individual base models.


🧾 7. Conclusions & Future Work
🧠 Key Takeaways:

Sentiment on Twitter during COVID-19 leaned strongly negative, especially in politically or health-related topics.

KMeans topic modeling effectively grouped tweets into meaningful themes, revealing how public concern shifted over time.

Voting-based classifiers (especially Hard Voting) yielded the best overall performance in sentiment prediction, with an F1-score of 0.80.

🚀 Future Improvements:

Incorporate transformer-based models (e.g., BERT) for deeper semantic understanding.

Use dynamic topic modeling to capture changes in themes over time.

Expand the dataset beyond COVID-19 for more generalizable sentiment trends.
