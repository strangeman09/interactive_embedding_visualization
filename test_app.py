import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px


def reduce_dimensions(data, method='pca', n_components=2):
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
    else:
        raise ValueError("Invalid method. Choose 'pca', 'tsne', or 'umap'.")
    
    return reducer.fit_transform(data)

with open('data.json', 'r') as json_file:
    data = json.load(json_file)

# Convert lists back to NumPy arrays and create a DataFrame
for item in data:
    item["embeddings"] = np.array(item["embeddings"])

df = pd.DataFrame(data)



# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout of the app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Interactive Embedding Visualization"),
            dcc.Dropdown(
                id='method-dropdown',
                options=[
                    {'label': 'PCA', 'value': 'pca'},
                    {'label': 't-SNE', 'value': 'tsne'},
                    {'label': 'UMAP', 'value': 'umap'}
                ],
                value='pca',
                clearable=False
            ),
            dcc.Dropdown(
                id='dimension-dropdown',
                options=[
                    {'label': '2D', 'value': 2},
                    {'label': '3D', 'value': 3}
                ],
                value=2,
                clearable=False
            ),
            dcc.Graph(id='embedding-graph')
        ])
    ])
])


@app.callback(
    Output('embedding-graph', 'figure'),
    [Input('method-dropdown', 'value'),
     Input('dimension-dropdown', 'value')]
)
def update_graph(method, n_components):
    reduced_data = reduce_dimensions(df['embeddings'].to_list(), method=method, n_components=n_components)
    if n_components == 2:
        fig = px.scatter(
            x=reduced_data[:, 0], y=reduced_data[:, 1], color=df['class_label'],
            hover_data={'file_path': df['file_path'], 'embeddings': df['embeddings']}
        )
    else:
        fig = px.scatter_3d(
            x=reduced_data[:, 0], y=reduced_data[:, 1], z=reduced_data[:, 2], color=df['class_label'],
            hover_data={'file_path': df['file_path'], 'embeddings': df['embeddings']}
        )
    
    fig.update_traces(marker=dict(size=5, opacity=0.8),
                      selector=dict(mode='markers'))
    
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
