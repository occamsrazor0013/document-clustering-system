import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import rand_score

def read_data(data_dir):
    documents = []
    labels = []
    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if os.path.isdir(category_path):
            for file in os.listdir(category_path):
                file_path = os.path.join(category_path, file)
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                    documents.append(content)
                    labels.append(category)
    return documents, labels

def calculate_rand_index(true_labels, predicted_labels):
    print(f'Rand Index: {rand_score(true_labels, predicted_labels)}')

def top_terms_per_cluster(X, labels, vectorizer, k):
    for cluster_id in range(k):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_terms_tfidf = np.sum(X[cluster_indices].toarray(), axis=0)
        top_terms_indices = cluster_terms_tfidf.argsort()[-5:][::-1]
        top_terms = [vectorizer.get_feature_names_out()[i] for i in top_terms_indices]
        print(f"Top 5 terms in Cluster {cluster_id+1}: {', '.join(top_terms)}")

def write_clusters_to_file(k, clusters, labels, documents):
    with open('cluster_information.txt', 'w') as file:
        for cluster_num in range(k):
            cluster_indices = np.where(clusters == cluster_num)[0]
            file.write(f'Cluster {cluster_num + 1}:\n\n')
            for index in cluster_indices:
                file.write(f'Document ID: {index + 1}\n')
                file.write(f'Title: {documents[index].splitlines()[0]}\n')
                file.write(f'Actual Class Label: {labels[index]}\n\n')

def plot_clusters(X_reduced, labels, centroids):
    plt.figure(figsize=(11, 11))
    unique_labels = np.unique(labels)
    cluster = ['sport', 'business', 'politics', '' , 'tech'] 
    for cluster_label in unique_labels:
        cluster_points = X_reduced[labels == cluster_label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_label+1}', s=3)
    plt.scatter(centroids[:, 0], centroids[: , 1], marker="*", s=50, c='black')
    plt.title('KMeans Clustering of BBC News Documents')
    plt.legend()
    plt.show()

def main():
    k = 5

    data_dir = "bbc"
    documents, true_labels = read_data(data_dir)

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X.toarray())

    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit(X_reduced)
    centroids = kmeans.cluster_centers_
    pred_labels = kmeans.labels_

    calculate_rand_index(true_labels, pred_labels)

    top_terms_per_cluster(X, pred_labels, vectorizer, k)

    write_clusters_to_file(k, pred_labels, true_labels, documents)

    plot_clusters(X_reduced, pred_labels, centroids)

if __name__ == "__main__":
    main()