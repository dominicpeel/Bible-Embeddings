# dimensionality reduction using t-SNE
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import ast

# load data
df = pd.read_csv('data/embeddings.csv')
mat = np.array(df['embedding'].apply(ast.literal_eval).tolist())

# dimensionality reduction
tsne = TSNE(n_components=3)
red_dims = tsne.fit_transform(mat)

# create dataframe with reduced dimensions
cols=['book', 'chapter', 'verses', 'content', 'x', 'y', 'z', 'nn']
df_red = pd.DataFrame(columns=cols)

df_red['book'] = df['book']
df_red['chapter'] = df['chapter']
df_red['verses'] = df['verses']
df_red['content'] = df['content']

df_red['x'] = red_dims[:, 0]
df_red['y'] = red_dims[:, 1]

# find 5 nearest neighbors
for i in range(len(df_red)):
    sim = cosine_similarity(mat[i].reshape(1, -1), mat)
    sim = sim.reshape(-1)
    sim[i] = 0
    nn = np.argsort(sim)[-5:]
    df_red['nn'][i] = nn

df_red.head()
print(df_red['nn'][0])



