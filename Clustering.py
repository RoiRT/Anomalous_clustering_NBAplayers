import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
from sklearn.cluster import KMeans

mpl.use('Qt5Agg')

df_RS = pd.read_csv('2022-2023 NBA Player Stats - Regular.csv', delimiter=";", encoding="latin-1")
df_PO = pd.read_csv('2022-2023 NBA Player Stats - Playoffs.csv', delimiter=";", encoding="latin-1")

df_RS_ = pd.merge(df_PO, df_RS , on=['Player', 'Tm'], how='inner').filter(regex='^(?!.*_x$)')
df_PO_ = pd.merge(df_PO, df_RS_ , on=['Player', 'Tm'], how='inner').filter(regex='^(?!.*_y$)')

RS_columns = {col: col.split('_')[0] for col in df_RS_.columns}
PO_columns = {col: col.split('_')[0] for col in df_PO_.columns}

df_RS_.rename(columns=RS_columns, inplace=True)
df_PO_.rename(columns=PO_columns, inplace=True)

set(df_PO_.columns) == set(df_RS_.columns)

df_dif = df_PO_.copy()
cols = df_dif.columns.values
df_dif[cols[7:]] = df_dif[cols[7:]] - df_RS_[cols[7:]]

scaler = MinMaxScaler(feature_range=(-1, 1))
df_dif[list(cols[7:])+['Age']] = scaler.fit_transform(df_dif[list(cols[7:])+['Age']])



a = 'PTS'
b = 'Age'

X = df_dif[[a, b]].values
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Obtener las etiquetas de los clusters y los centroides
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

fig, ax = plt.subplots()

scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
ax.legend(*scatter.legend_elements(), title="Clusters")
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200)
plt.axvline(x=0, color='red', linestyle='--')
plt.xlabel(a)
plt.ylabel(b)
plt.title('K-Means Clustering')
plt.show()