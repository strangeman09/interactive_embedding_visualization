import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import base64
import dash
from dash import dcc, html, Input, Output, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
import argparse


def encode_image(image_file):
    with open(image_file, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')


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



parser = argparse.ArgumentParser(description="Interactive Embedding Visualization")
parser.add_argument('data_path', type=str, help='Path to the JSON file containing embeddings data')
args = parser.parse_args()


with open(args.data_path, 'r') as json_file:
    data = json.load(json_file)


for item in data:
    item["embeddings"] = np.array(item["embeddings"])

df = pd.DataFrame(data)
df['image'] = df['file_path'].apply(encode_image)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


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
            dcc.Graph(id='embedding-graph'),
            dcc.Tooltip(id="graph-tooltip", direction='bottom')
        ])
    ])
])


@app.callback(
    Output('embedding-graph', 'figure'),
    [Input('method-dropdown', 'value'),
     Input('dimension-dropdown', 'value')]
)
def update_graph(method, n_components):
    embeddings = np.array(df['embeddings'].tolist())
    reduced_data = reduce_dimensions(embeddings, method=method, n_components=n_components)
    reduced_df = pd.DataFrame(reduced_data, columns=[f'Component {i+1}' for i in range(n_components)])
    reduced_df['class_label'] = df['class_label']
    reduced_df['image'] = df['image']
    
    if n_components == 2:
        fig = px.scatter(
            reduced_df, x='Component 1', y='Component 2', color='class_label',
            custom_data=['image', 'class_label']
        )
    else:
        fig = px.scatter_3d(
            reduced_df, x='Component 1', y='Component 2', z='Component 3', color='class_label',
            custom_data=['image', 'class_label']
        )

    hover_template = """
    <b>Class: %{customdata[1]}</b><br><br>
    <img src='data:image/png;base64,%{customdata[0]}' width='150'><br>
    """
    fig.update_traces(marker=dict(size=10, opacity=0.8),
                      selector=dict(mode='markers'),
                      hovertemplate=hover_template)
    fig.update_layout(height=800, width=1500)
    
    return fig


@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("embedding-graph", "hoverData")
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update


    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]

  
    image_base64 = hover_data["customdata"][0]
    class_label = hover_data["customdata"][1]

    children = [
        html.Div([
            html.Img(
                src='data:image/png;base64,{}'.format(image_base64),
                style={"width": "150px", 'display': 'block', 'margin': '0 auto'},
            ),
            html.P("Class: " + str(class_label), style={'font-weight': 'bold'})
        ])
    ]

    return True, bbox, children


if __name__ == '__main__':
    app.run_server(debug=True)
