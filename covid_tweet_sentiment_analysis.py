import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import nltk
from nltk.corpus import stopwords
import re
import tensorflow as tf
import contractions
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, \
    f1_score  # Метрики за оценка
from urlextract import URLExtract
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from time import time
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ------------ Импортиране на библиотеки за BERTopic ------------
# from bertopic import BERTopic
# import matplotlib.pyplot as plt


train = pd.read_csv(
    r'C:\Users\Sergey Filipov\Desktop\NLP-Final\NLP_train_test\Corona_NLP_train.csv', encoding='latin')
test = pd.read_csv(
    r'C:\Users\Sergey Filipov\Desktop\NLP-Final\NLP_train_test\Corona_NLP_test.csv', encoding='latin')


# Добавяне на колона, която указва дали редът е от тренировъчния или тестовия набор
train['split'] = 'train'
test['split'] = 'test'


# Обединяване на тренировъчните и тестовите данни в един DataFrame
df = pd.concat([train, test], axis=0)
df.reset_index(drop=True, inplace=True)  # Нулиране на индексите за последователност


# Проверка и премахване на дублирани редове
duplicates = df.duplicated()
print(f'Брой дублирани редове: {duplicates.sum()}')
df = df.drop_duplicates()  # Премахване на дублираните записи


# Проверка за липсващи стойности във всяка колона
missing_values = df.isna().sum()
print('Брой липсващи стойности във всяка колона:\n', missing_values)
df['Location'] = df['Location'].fillna('Unknown')  # Заместване на липсващи стойности в колоната Location с "Unknown"
missing_values_after = df.isna().sum()  # Проверка за липсващи стойности след замяната
print('Брой липсващи стойности във всяка колона след замяна:\n', missing_values_after)

# Записване на резултатите в Excel файл
# output_file = r'C:\Users\Sergey Filipov\Desktop\final_locations.xlsx'
# df.to_excel(output_file, index=False)
# print(f"Файлът с коригираните локации е записан в {output_file}")

# Изтегляне на списъка със стоп думи (думи без съществена стойност за анализа)
nltk.download('stopwords')
stopwords_list = stopwords.words('english')
extractor = URLExtract()  # Обект за извличане на URL адреси от текст


# Функция за предварителна обработка на текстовете (туитове)
def prepare_tweet(tweet):
    urls = extractor.find_urls(tweet)  # Извличане на всички URL адреси в текста
    for u in urls:
        tweet = tweet.replace(u, "")  # Премахване на URL адресите

    tweet = contractions.fix(tweet)  # Разширяване на съкращенията (напр. can't -> cannot)
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(tweet)  # Токенизация текста на отделни думи
    filtered_tokens = []

    for t in tokens:
        if not re.match("[a-zA-Z]+", t):  # Игнориране на символи, които не са букви
            continue
        if t.lower() in stopwords_list:  # Премахване на стоп думи
            continue
        filtered_tokens.append(t)

    return filtered_tokens  # Връщане на почистените и токенизирани думи


# Приложение на функцията за обработка върху колоната с оригинални туитове
df['TokenizedTweet'] = df['OriginalTweet'].apply(prepare_tweet)

# Премахване на редове с празни токени (където всички думи са били премахнати)
mask = df['TokenizedTweet'].str.len() == 0
df = df[~mask]  # Оставяме само редове, които имат съдържание

# Преобразуване на категориите на настроенията в числови стойности за обучение
sentiment_map = {'Extremely Negative': 0, 'Negative': 0, 'Neutral': 1, 'Positive': 2, 'Extremely Positive': 2}
y = df['Sentiment'].map(sentiment_map)

# Разделяне на данните обратно на тренировъчни и тестови чрез маски
train_mask = df['split'] == 'train'
test_mask = ~train_mask

# Векторизация на текста чрез TF-IDF, която отразява важността на думите в контекста на документа
vectorizer_tfidf = TfidfVectorizer()
X_train = vectorizer_tfidf.fit_transform(df['TokenizedTweet'][train_mask].apply(' '.join))
X_test = vectorizer_tfidf.transform(df['TokenizedTweet'][test_mask].apply(' '.join))


# Анализ на разпределението на датите на туитовете в обединените данни
dates_distribution = df['TweetAt'].value_counts().sort_index()
# print(dates_distribution.sort_values(ascending=False).head(5))

# Създаване на графиката с най-силните дати отбелязани на върха
plt.figure(figsize=(12, 6))
plt.plot(dates_distribution.index, dates_distribution.values, label='Обединени данни', color='green', marker='o')

# Търсене на локални върхове само с най-високата точка в близост
local_maxima = dates_distribution[(dates_distribution.diff().shift(-1) < 0) & (dates_distribution.diff() > 0)]

for date, count in local_maxima.items():  # Добавяне на текст само с броя туитове на върховите дати
    plt.text(date, count + 50, f'{count}', ha='center', va='bottom', fontsize=9, color='black')

# Форматиране оста X с по-добра четимост
plt.xticks(rotation=45, ha='right')
plt.title('Разпределение на туитовете по дати (обединени данни)')
plt.xlabel('Дата')
plt.ylabel('Брой туитове')
plt.legend()
plt.tight_layout()  # За да предотвратим рязане на осите
plt.show()  # Показване на графиката

del train, test  # Изтриване на оригиналните таблици за оптимизация на паметта


sentiment_counts = df['Sentiment'].value_counts()  # Преглед на броя на различните категории настроения в данните
# Визуализация на разпределението на настроенията
plt.figure(figsize=(8, 5))
bars = plt.bar(sentiment_counts.index, sentiment_counts.values, color='skyblue')

# Добавяне на точните числа върху стълбчетата
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom', fontsize=10)

plt.title('Разпределение на настроенията')
plt.xlabel('Категории на настроенията')
plt.ylabel('Брой на срещанията')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ------------ KMeans Клъстеризация ------------
print("Автоматично определяне на оптималния брой клъстери...")
sil_scores = []
K_values = range(2, 11)  # Ще тестваме от 2 до 10 клъстера

for k in K_values:
    print(f" - Тестване на {k} клъстера...")
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_train)  # Използваме правилната променлива X_train
    sil_score = silhouette_score(X_train, labels)
    sil_scores.append(sil_score)

# Намиране на оптималния брой клъстери
optimal_k = K_values[sil_scores.index(max(sil_scores))]
print(f"Оптимален брой клъстери: {optimal_k}")

print("Обучение на KMeans с оптималния брой клъстери...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df.loc[train_mask, 'Cluster'] = kmeans.fit_predict(X_train)


# ------------ Извличане на ключови думи за всяка тема ------------
def get_top_keywords(cluster, n_terms=5):
    cluster_tweets = df.loc[df['Cluster'] == cluster, 'OriginalTweet'].fillna('').tolist()

    # Проверка дали има туитове в клъстера
    if len(cluster_tweets) == 0:
        print(f"Клъстер {cluster} е празен. Пропускане...")
        return "Няма ключови думи"

    print(f" - Обработка на клъстер {cluster} с {len(cluster_tweets)} туитове...")

    cluster_text = ' '.join(cluster_tweets)

    # Списък с допълнителни стоп думи
    additional_stopwords = {'https', 'coronavirus', 'covid', '19', 'rt', 'amp'}

    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([cluster_text])

    # Изчисляваме броя на думите
    word_counts = X.sum(axis=0).A1
    keywords = [(word, word_counts[idx]) for word, idx in vectorizer.vocabulary_.items()]

    # Филтрираме ключови думи, които са в списъка със стоп думи
    filtered_keywords = [word for word, _ in sorted(keywords, key=lambda x: x[1], reverse=True)
                         if word not in additional_stopwords]

    # Ограничаваме до първите n_terms ключови думи
    return "\n".join(filtered_keywords[:n_terms])


print("6. Извличане на ключови думи за всеки клъстер...")
kmeans_keywords = {cluster: get_top_keywords(cluster) for cluster in range(optimal_k)}


def generate_topic_labels(kmeans_keywords):
    used_labels = set()  # Сет за следене на използваните имена на темите
    topic_labels = []

    for cluster, keywords in kmeans_keywords.items():
        # Вземаме първите 2 ключови думи като предложение за име на темата
        potential_label = keywords.split("\n")[:2]
        label = " ".join(potential_label).strip()

        # Проверка за уникалност на името на темата
        base_label = label
        suffix = 1
        while label in used_labels:
            # Ако има повторение, добавяме числов суфикс
            label = f"{base_label} {suffix}"
            suffix += 1

        used_labels.add(label)
        topic_labels.append(label)

    return topic_labels

# Стъпка 6: Извличане на ключови думи за всеки клъстер
print("6. Извличане на ключови думи за всеки клъстер...")
kmeans_keywords = {cluster: get_top_keywords(cluster) for cluster in range(optimal_k)}

# Генериране на уникални имена на темите
topic_labels = generate_topic_labels(kmeans_keywords)

# Генериране на уникални имена на темите
topic_labels = generate_topic_labels(kmeans_keywords)

# Стъпка 7: Визуализация на броя на туитовете с имената на темите върху стълбовете
print("7. Визуализация на броя на туитовете и имената на темите...")
kmeans_counts = df['Cluster'].value_counts().reset_index()
kmeans_counts.columns = ['Тема', 'Брой туитове']

plt.figure(figsize=(22, 16))  # Увеличен размер за повече място
bars = sns.barplot(x=topic_labels, y='Брой туитове', data=kmeans_counts)

# Добавяме броя на туитовете върху стълбчетата с оптимизирано разстояние
for bar, (topic, count) in zip(bars.patches, kmeans_counts.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 400, f'{int(count)}',
             ha='center', va='bottom', fontsize=14, fontweight='bold')

# Разширяваме полето за заглавието и увеличаваме шрифта
plt.title('Теми в туитовете (KMeans)', fontsize=22, pad=60)  # Увеличен padding за заглавието
plt.xlabel('Теми', fontsize=16)
plt.ylabel('Брой туитове', fontsize=16)

# Значително разширяваме пространството под етикетите на X оста
plt.xticks(rotation=35, ha='right', fontsize=14)
plt.subplots_adjust(bottom=0.7, top=0.77)  # Още повече разстояние долу и горе

# Увеличаваме лимита на оста Y за повече пространство над числата
max_count = kmeans_counts['Брой туитове'].max()
plt.ylim(0, max_count + 3000)  # Оставяме повече място над стълбчетата

plt.tight_layout()  # Оптимизация за избягване на припокриване
plt.show()




# Персонализирани цветове за различните настроения (адаптирани към реалните стойности)
custom_colors = {
    'Extremely Negative': '#8B0000',  # Тъмно червено
    'Negative': '#FF0000',            # Червено
    'Neutral': '#1E90FF',             # Синьо
    'Positive': '#ADFF2F',            # Светло зелено
    'Extremely Positive': '#006400'   # Тъмно зелено
}

# Групиране на туитовете по клъстери и изчисляване на разпределението на настроенията
sentiment_distribution = df.groupby('Cluster')['Sentiment'].value_counts(normalize=True).unstack().fillna(0)

# Визуализация на разпределението на настроенията
plt.figure(figsize=(18, 10))
sentiment_distribution.plot(kind='bar', stacked=True,
                            color=[custom_colors[col] for col in sentiment_distribution.columns if col in custom_colors])

# Настройка на заглавието и осите
plt.title('Разпределение на настроенията по теми (автоматично извлечени чрез KMeans)', fontsize=18)
plt.xlabel('Теми', fontsize=14)
plt.ylabel('Процентно разпределение на настроенията', fontsize=14)
plt.legend(title='Настроение', bbox_to_anchor=(1.15, 1), loc='upper left')

# Използваме реалните имена на темите от първата графика (topic_labels)
plt.xticks(ticks=range(len(topic_labels)), labels=topic_labels, rotation=30, ha='right', fontsize=12)

plt.tight_layout()  # За да избегнем припокриване на елементи
plt.show()


# # ------------ Извличане на теми с BERTopic ------------
# # Извличане на теми с BERTopic
# topic_model = BERTopic()
# topics, _ = topic_model.fit_transform(df['TokenizedTweet'][train_mask].apply(' '.join))

# # Проверка на броя туитове в шум (тема -1)
# num_noise_tweets = (pd.Series(topics) == -1).sum()
# print(f"Брой туитове, класифицирани като шум: {num_noise_tweets}")

# # Честота на темите (включително шума)
# topic_counts = pd.DataFrame(topic_model.get_topic_freq())
# topic_counts.columns = ['Тема', 'Брой туитове']

# # Премахване на шума (остави това ако искаш само основните теми)
# topic_counts = topic_counts[topic_counts['Тема'] != -1]

# # Проверка на общия брой туитове
# total_processed_tweets = topic_counts['Брой туитове'].sum() + num_noise_tweets
# print(f"Общ брой обработени туитове: {total_processed_tweets}")
# print(f"Общ брой туитове в набора: {len(df[train_mask])}")

# # Извличане на съкратени ключови думи за всяка тема от BERTopic
# topics_data = topic_model.get_topics()
# bertopic_keywords = {topic: ", ".join([word for word, _ in words[:3]]) for topic, words in topics_data.items() if topic != -1}

# # Генериране на обобщено име за всяка тема (на базата на ключовите думи)
# def generate_topic_label(topic):
#     keywords = bertopic_keywords[topic].split(', ')
#     return f"{keywords[0]} & {keywords[1]}" if len(keywords) >= 2 else keywords[0]

# # ------------ Визуализация на топ 7 теми ------------
# plt.figure(figsize=(18, 12))  # Голям размер за четливост
# # Ограничаваме се до топ 7 теми
# top_7_topic_counts = topic_counts.nlargest(7, 'Брой туитове')
# topic_labels = [generate_topic_label(topic) for topic in top_7_topic_counts['Тема']]
# topic_numbers = [f"Тема {i+1}" for i in range(len(topic_labels))]  # Добавяме номера на темите

# bars = sns.barplot(x=topic_numbers, y='Брой туитове', data=top_7_topic_counts, palette='viridis')

# # Показване на ключовите думи над стълбовете
# for bar, (topic, count) in zip(bars.patches, top_7_topic_counts.values):
#     label = generate_topic_label(topic)
#     plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 150, label, ha='center', va='bottom', fontsize=12)

# plt.ylim(0, 2200)  # Лимит на оста Y
# plt.title('Топ 7 теми в туитовете (BERTopic)', fontsize=16)
# plt.xlabel('Теми', fontsize=14)
# plt.ylabel('Брой туитове', fontsize=14)
# plt.xticks(rotation=0)
# plt.show()
# topic_model.visualize_hierarchy()


# Функция за измерване на времето за обучение и прогнозиране на моделите
def time_model(model, X_train, y_train, X_test, n_iter=10):
    st_fit = time()  # Начало на измерване на времето за обучение
    for _ in range(n_iter - 1):
        clone(model).fit(X_train, y_train)  # Обучение на клонирания модел
    model = clone(model)
    model.fit(X_train, y_train)  # Последно обучение на оригиналния модел
    et_fit = time()  # Край на измерването на времето за обучение

    st_pred = time()  # Начало на измерването на времето за прогнозиране
    for _ in range(n_iter):
        pred = model.predict(X_test)  # Прогнозиране върху тестовите данни
    et_pred = time()  # Край на измерването на времето за прогнозиране

    return pred, (et_fit - st_fit) / n_iter, (et_pred - st_pred) / n_iter  # Връщане на резултатите


# Клас за персонализиран класификатор на базата на Bernoulli Naive Bayes
class BernoulliNBWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = BernoulliNB()
        self.transform = lambda X: (X > 0).astype(int)  # Бинаризиране на входните данни

    def fit(self, X, y):
        self.model.fit(self.transform(X), y)  # Обучение на модела с преобразувани данни
        return self

    def predict(self, X):
        return self.model.predict(self.transform(X))  # Прогнозиране с преобразувани данни

    def predict_proba(self, X):
        return self.model.predict_proba(self.transform(X))  # Вероятностни прогнози


# Обучение на базов модел Bernoulli Naive Bayes
model_bnb = BernoulliNB()
pred_bnb, time_bnb_fit, time_bnb_pred = time_model(model_bnb, X_train, y[train_mask], X_test)

# Обучение на модел Logistic Regression
model_lr = LogisticRegression(max_iter=1000)
pred_lr, time_lr_fit, time_lr_pred = time_model(model_lr, X_train, y[train_mask], X_test)

# Обучение на LinearSVC оптимизация чрез GridSearch
linear_lsvc = LinearSVC(dual=False, penalty='l2', loss='squared_hinge')
grid_search = GridSearchCV(linear_lsvc, param_grid={'C': [0.01, 0.1, 1, 10]}, cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y[train_mask])
model_lsvc = grid_search.best_estimator_
pred_lsvc, time_lsvc_fit, time_lsvc_pred = time_model(model_lsvc, X_train, y[train_mask], X_test)

# Създаване на ансамблови модели (Hard и Soft Voting)
hard_ensemble = VotingClassifier(estimators=[('bnb', model_bnb), ('lr', model_lr), ('lsvc', model_lsvc)], voting='hard')
pred_hard_ens, time_hard_ens_fit, time_hard_ens_pred = time_model(hard_ensemble, X_train, y[train_mask], X_test)

soft_ensemble = VotingClassifier(
    estimators=[('bnb', model_bnb), ('lr', model_lr), ('lsvc', CalibratedClassifierCV(model_lsvc))], voting='soft')
pred_soft_ens, time_soft_ens_fit, time_soft_ens_pred = time_model(soft_ensemble, X_train, y[train_mask], X_test)


# Функция за визуализация матрица на объркванията и извеждане на отчет
def confusion_matrix_classification_report(y, preds, labels):
    cm = confusion_matrix(y, preds)  # Изчисляване на матрицата на объркванията
    matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)  # Визуализация на резултатите
    matrix.plot(cmap='Blues')
    f1 = f1_score(y, preds, average='weighted')  # Изчисляване на F1-скор
    plt.title(f'F1 score {f1:.2f}')
    plt.show()
    return f1, classification_report(y, preds)


# Оценка на всеки модел чрез матрица на объркванията и F1-скор

# Оценка на модела Bernoulli Naive Bayes (BNB) # Изчислява F1-скор и отчета за метриките
f1bnb, rep = confusion_matrix_classification_report(y[test_mask], pred_bnb, ['Negative', 'Neutral', 'Positive'])
print(rep)  # Извеждане на отчета на екрана

# Оценка на модела Logistic Regression (LR) # Изчислява F1-скор и отчета за логистичната регресия
f1lr, rep = confusion_matrix_classification_report(y[test_mask], pred_lr, ['Negative', 'Neutral', 'Positive'])
print(rep)  # Извеждане на отчета

# Оценка на модела Linear Support Vector Classifier (LinearSVC)  # Оценка на модела SVM
f1lsvc, rep = confusion_matrix_classification_report(y[test_mask], pred_lsvc, ['Negative', 'Neutral', 'Positive'])
print(rep)  # Отчет за производителността

# Оценка на ансамбловия модел Hard Voting (твърдо гласуване) # Оценка на комбинирания модел чрез твърдо гласуване
f1_hard_ens, rep = (
    confusion_matrix_classification_report(y[test_mask], pred_hard_ens, ['Negative', 'Neutral', 'Positive']))
print(rep)  # Показване на резултатите

# Оценка на ансамбловия модел Soft Voting (меко гласуване) # Оценка чрез меко гласуване
f1_soft_ens, rep = (
    confusion_matrix_classification_report(y[test_mask], pred_soft_ens, ['Negative', 'Neutral', 'Positive']))
print(rep)  # Показване на резултатите

# Създаване на списъци с времената за обучение и прогнозиране
fit_times = [time_bnb_fit, time_lr_fit, time_lsvc_fit, time_hard_ens_fit, time_soft_ens_fit]
pred_times = [time_bnb_pred, time_lr_pred, time_lsvc_pred, time_hard_ens_pred, time_soft_ens_pred]

# Създаване на DataFrame за сравнение на моделите
res = pd.DataFrame({
    'Fit time (s)': fit_times,
    'Prediction time (s)': pred_times,
    'f1 score': [f1bnb, f1lr, f1lsvc, f1_hard_ens, f1_soft_ens],  # Колона с F1-скор за всеки модел
    '% increase relative to BNB fit time': [(x - time_bnb_fit) / time_bnb_fit * 100 for x in fit_times],
    '% increase relative to BNB pred time': [(x - time_bnb_pred) / time_bnb_fit * 100 for x in pred_times]
}, index=['BernoulliNB', 'LinearRegression', 'LinearSVC', 'Hard voting', 'Soft voting'])

# Сортиране на моделите по F1-скор в низходящ ред и показване на таблицата
res.sort_values(by='f1 score', ascending=False)

