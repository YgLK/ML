cluster_nums = [8,9,10,11,12]
silh_scores = []

for k in cluster_nums:
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_pred = kmeans.fit_predict(X)
    s_score = silhouette_score(X, kmeans.labels_)
    silh_scores.append(s_score)
    
print("Silhouette score list:", silh_scores)
