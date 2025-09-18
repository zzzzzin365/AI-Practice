import pandas as pd
video_data = pd.read_csv("origin_videos_data.csv")
comments_data = pd.read_csv("origin_comments_data.csv")
video_data.sample(10)
comments_data.head()
video_data["text"] = video_data["video_desc"].fillna("") + " " + video_data["video_tags"].fillna("")
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

product_name_predictor = make_pipeline(
    TfidfVectorizer(tokenizer=jieba.lcut, max_features=50), SGDClassifier()
)
product_name_predictor.fit(
    video_data[~video_data["product_name"].isnull()]["text"],
    video_data[~video_data["product_name"].isnull()]["product_name"],
)
video_data["product_name"] = product_name_predictor.predict(video_data["text"])
comments_data.columns
for col in ['sentiment_category',
       'user_scenario', 'user_question', 'user_suggestion']:
    predictor = make_pipeline(
        TfidfVectorizer(tokenizer=jieba.lcut), SGDClassifier()
    )
    predictor.fit(
        comments_data[~comments_data[col].isnull()]["comment_text"],
        comments_data[~comments_data[col].isnull()][col],
    )
    comments_data[col] = predictor.predict(comments_data["comment_text"])
    top_n_words = 10
    kmeans_predictor = make_pipeline(
    TfidfVectorizer(tokenizer=jieba.lcut), KMeans(n_clusters=2)
)

kmeans_predictor.fit(comments_data[comments_data["sentiment_category"].isin([1, 3])]["comment_text"])
kmeans_cluster_label = kmeans_predictor.predict(comments_data[comments_data["sentiment_category"].isin([1, 3])]["comment_text"])

kmeans_top_word = []
tfidf_vectorizer = kmeans_predictor.named_steps['tfidfvectorizer']
kmeans_model = kmeans_predictor.named_steps['kmeans']
feature_names = tfidf_vectorizer.get_feature_names_out()
cluster_centers = kmeans_model.cluster_centers_
for i in range(kmeans_model.n_clusters):
    top_feature_indices = cluster_centers[i].argsort()[::-1]
    top_word = ' '.join([feature_names[idx] for idx in top_feature_indices[:top_n_words]])
    kmeans_top_word.append(top_word)

comments_data.loc[comments_data["sentiment_category"].isin([1, 3]), "positive_cluster_theme"] = [kmeans_top_word[x] for x in kmeans_cluster_label]
kmeans_predictor = make_pipeline(
    TfidfVectorizer(tokenizer=jieba.lcut), KMeans(n_clusters=2)
)

kmeans_predictor.fit(comments_data[comments_data["sentiment_category"].isin([2, 3])]["comment_text"])
kmeans_cluster_label = kmeans_predictor.predict(comments_data[comments_data["sentiment_category"].isin([2, 3])]["comment_text"])

kmeans_top_word = []
tfidf_vectorizer = kmeans_predictor.named_steps['tfidfvectorizer']
kmeans_model = kmeans_predictor.named_steps['kmeans']
feature_names = tfidf_vectorizer.get_feature_names_out()
cluster_centers = kmeans_model.cluster_centers_
for i in range(kmeans_model.n_clusters):
    top_feature_indices = cluster_centers[i].argsort()[::-1]
    top_word = ' '.join([feature_names[idx] for idx in top_feature_indices[:top_n_words]])
    kmeans_top_word.append(top_word)

comments_data.loc[comments_data["sentiment_category"].isin([2, 3]), "negative_cluster_theme"] = [kmeans_top_word[x] for x in kmeans_cluster_label]
kmeans_predictor = make_pipeline(
    TfidfVectorizer(tokenizer=jieba.lcut), KMeans(n_clusters=2)
)

kmeans_predictor.fit(comments_data[comments_data["user_scenario"].isin([1])]["comment_text"])
kmeans_cluster_label = kmeans_predictor.predict(comments_data[comments_data["user_scenario"].isin([1])]["comment_text"])

kmeans_top_word = []
tfidf_vectorizer = kmeans_predictor.named_steps['tfidfvectorizer']
kmeans_model = kmeans_predictor.named_steps['kmeans']
feature_names = tfidf_vectorizer.get_feature_names_out()
cluster_centers = kmeans_model.cluster_centers_
for i in range(kmeans_model.n_clusters):
    top_feature_indices = cluster_centers[i].argsort()[::-1]
    top_word = ' '.join([feature_names[idx] for idx in top_feature_indices[:top_n_words]])
    kmeans_top_word.append(top_word)

comments_data.loc[comments_data["user_scenario"].isin([1]), "scenario_cluster_theme"] = [kmeans_top_word[x] for x in kmeans_cluster_label]
kmeans_predictor = make_pipeline(
    TfidfVectorizer(tokenizer=jieba.lcut), KMeans(n_clusters=2)
)

kmeans_predictor.fit(comments_data[comments_data["user_question"].isin([1])]["comment_text"])
kmeans_cluster_label = kmeans_predictor.predict(comments_data[comments_data["user_question"].isin([1])]["comment_text"])

kmeans_top_word = []
tfidf_vectorizer = kmeans_predictor.named_steps['tfidfvectorizer']
kmeans_model = kmeans_predictor.named_steps['kmeans']
feature_names = tfidf_vectorizer.get_feature_names_out()
cluster_centers = kmeans_model.cluster_centers_
for i in range(kmeans_model.n_clusters):
    top_feature_indices = cluster_centers[i].argsort()[::-1]
    top_word = ' '.join([feature_names[idx] for idx in top_feature_indices[:top_n_words]])
    kmeans_top_word.append(top_word)

comments_data.loc[comments_data["user_question"].isin([1]), "question_cluster_theme"] = [kmeans_top_word[x] for x in kmeans_cluster_label]
kmeans_predictor = make_pipeline(
    TfidfVectorizer(tokenizer=jieba.lcut), KMeans(n_clusters=2)
)

kmeans_predictor.fit(comments_data[comments_data["user_suggestion"].isin([1])]["comment_text"])
kmeans_cluster_label = kmeans_predictor.predict(comments_data[comments_data["user_suggestion"].isin([1])]["comment_text"])

kmeans_top_word = []
tfidf_vectorizer = kmeans_predictor.named_steps['tfidfvectorizer']
kmeans_model = kmeans_predictor.named_steps['kmeans']
feature_names = tfidf_vectorizer.get_feature_names_out()
cluster_centers = kmeans_model.cluster_centers_
for i in range(kmeans_model.n_clusters):
    top_feature_indices = cluster_centers[i].argsort()[::-1]
    top_word = ' '.join([feature_names[idx] for idx in top_feature_indices[:top_n_words]])
    kmeans_top_word.append(top_word)

comments_data.loc[comments_data["user_suggestion"].isin([1]), "suggestion_cluster_theme"] = [kmeans_top_word[x] for x in kmeans_cluster_label]
