from itertools import chain
import logging
import os

import numpy as np

from src.calculations.mortality_tables import project_lc, read_lc, per2gen
from src.calculations.present_values import *
from src.calculations.uws import Uws

logger =logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

ages = list(range(60, 105))
years = list(range(1960, 2016))
gens = list(range(1940, 1991))

base_dir = os.path.dirname(__file__)
# base_dir = "."

print(os.getcwd())
ex = pd.read_table(os.path.join(base_dir, "data/ex.csv"), sep=",").set_index("Age")
ey = pd.read_table(os.path.join(base_dir, "data/ey.csv"), sep=",").set_index("Age")
dx = pd.read_table(os.path.join(base_dir, "data/dx.csv"), sep=",").set_index("Age")
dy = pd.read_table(os.path.join(base_dir, "data/dy.csv"), sep=",").set_index("Age")
mx = dx / ex
mx.columns = map(int, mx.columns)
mx.fillna(999, inplace=True)
my = dy / ey
my.columns = map(int, my.columns)
my.fillna(999, inplace=True)

lc_x = read_lc(os.path.join(base_dir, "data/lc_x.csv"))
lc_y = read_lc(os.path.join(base_dir, "data/lc_y.csv"))
qx_proj = project_lc(mx.loc[list(map(int, lc_x["x"])), :], lc_x)
qy_proj = project_lc(my.loc[list(map(int, lc_y["x"])), :], lc_y)

i = pd.read_table(os.path.join(base_dir, "data/i_7_rolling.csv"), sep=",")["x"]
i_pre = pd.Series([i.iloc[0]]*200, index=list(range(i.index[0] - 200, i.index[0])))
i_post = pd.Series([i.iloc[-1]]*300, index=list(range(2032, 2332)))
i_proj = i_pre.append(i, verify_integrity=True).append(i_post, verify_integrity=True) / 100

uws = Uws(per2gen(qx_proj).loc[:, gens], per2gen(qy_proj).loc[:, gens], i_proj, {"k": 0.02, "m": 4}, {"wx": 0.6, "wy": 0, "kx": 0, "ky": 0.2})


# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Mortality Projection of a 65 year old male'),

    html.Div(children='''
        Projection using the Lee Carter Model. wheeeee
    '''),

    dcc.Slider(
        id="force_slider",
        min=-1,
        max=3,
        step=0.1,
        value=1,
        marks={i: '{}'.format(i) for i in range(-1, 10)},
    ),
    html.Div(children=".", style={"height": "150 pt"}),
    dcc.Graph(
        id='mortalities_x',
    ),
    dcc.Graph(
        id='mortalities_y',
    ),
])


@app.callback(
    dash.dependencies.Output('mortalities_x', 'figure'),
    [dash.dependencies.Input('force_slider', 'value')])
def update_output(force):
    qx_proj2 = project_lc(mx.loc[list(map(int, lc_x["x"])), :], lc_x, force)
    old = list(range(min(qx_proj2.columns), 2017))
    new = list(range(2017, max(qx_proj2.columns + 1)))
    ages = [60, 70, 80, 90]

    tick_loc = [np.log10(number) for number in [0.05, 0.01, 0.005, 0.001, 0.0005]]
    layout = {
        "yaxis": {
            "tickvals": tick_loc,
            "ticktext": [str(np.round(10 ** number * 100, 2)) + "%" for number in tick_loc]
        }
    }

    return {
        'data': list(chain(*[[
            {
                'type': 'scatter',
                "x": old,
                'y': np.log10(qx_proj2.loc[i, old]),
                "name": "Observed"
            },
            {
                'type': 'scatter',
                "x": new,
                'y': np.log10(qx_proj2.loc[i, new]),
                "name": "Projected"
            }] for i in ages])),
        'layout': {
            'title': "Male mortalities",
            "layout": layout
        }
    }


@app.callback(
    dash.dependencies.Output('mortalities_y', 'figure'),
    [dash.dependencies.Input('force_slider', 'value')])
def update_output(force):
    qy_proj2 = project_lc(my.loc[list(map(int, lc_x["x"])), :], lc_y, force)
    old = list(range(min(qy_proj2.columns), 2017))
    new = list(range(2017, max(qy_proj2.columns + 1)))
    ages = [60, 70, 80, 90]
    return {
        'data': list(chain(*[[
            {
                'type': 'scatter',
                "x": old,
                'y': qy_proj2.loc[i, old],
                "name": "Observed"
            },
            {
                'type': 'scatter',
                "x": new,
                'y': qy_proj2.loc[i, new],
                "name": "Projected"
            }] for i in ages])),
        'layout': {
            'title': "Female mortalities"
        }
    }


if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0")
