import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import ast
import plotly.graph_objs as go
import plotly.express as px

df = pd.read_csv('data/embeddings.csv')
# book,chapter,verses,content,embedding

books = df['book'].unique()
embeddings = np.array(df['embedding'].apply(ast.literal_eval).tolist())

# dimensionality reduction
tsne = TSNE(n_components=3)
red_dims = tsne.fit_transform(embeddings)

# Assuming 'vis_dims_3d' is the result of t-SNE in 3D, and 'chapters' is the dictionary with chapter information
x_3d = [x for x, y, z in red_dims]
y_3d = [y for x, y, z in red_dims]
z_3d = [z for x, y, z in red_dims]

# Create a color scale based on the book index
book_indices = {book: idx for idx, book in enumerate(books)}
colors_list = [book_indices[book] for book in df['book']]
color_scale = px.colors.diverging.Picnic
# color_scale = px.colors.qualitative.Plotly

# Create a list of custom hover texts
hover_texts = [f"{book} {chapter}" for book, chapter in zip(df['book'], df['chapter'])]

# Create a 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=x_3d,
    y=y_3d,
    z=z_3d,
    mode='markers',
    marker=dict(
        size=4,
        color=colors_list,
        colorscale=color_scale,
        showscale=True
    ),
    text=hover_texts,
    hovertemplate="%{text}<extra></extra>",
)])

fig.update_layout(
    title="Bible chapters embedded in 3D space using t-SNE",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    ),
    showlegend=False
)

fig.show()
