import json
from dash_html_components.Audio import Audio
from numpy.core.fromnumeric import repeat

import pandas as pd
from pathlib import Path

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output

THIS_DIR = Path(__file__).parent

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

with open(THIS_DIR / "data" / "ae_samples.json", "r") as fp:
    data = json.load(fp)

df = pd.DataFrame.from_dict(data)

fig = px.scatter(
    df,
    x="tsne_0",
    y="tsne_1",
    hover_name="path",
    hover_data={"tsne_0": False, "tsne_1": False},
)
fig.update_layout(xaxis_title="", yaxis_title="")

app.layout = html.Div(
    [
        html.H1("Sample Mapper"),
        html.Div(id="filename-output", children="", style={"height": "2vh"}),
        html.Audio(id="audio-player", autoPlay=True),
        dcc.Graph(id="tsne-plot", figure=fig, style={"height": "80vh"}),
    ],
)

@app.callback(
    Output("audio-player", "src"),
    Output("filename-output", "children"),
    Input("tsne-plot", "clickData")
)
def set_audio_file(clickData):
    try:
        filename = clickData["points"][0]["hovertext"]
        return f"/assets/{filename}", filename
    except:
        return "", ""


if __name__ == "__main__":
    app.run_server(host="0.0.0.0")
