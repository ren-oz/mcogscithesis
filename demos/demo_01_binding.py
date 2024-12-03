import numpy as np
import plotly.graph_objects as go
from dash import Dash, html, Input, Output, ctx, State, dcc

def get_random_points(n_s, n_u, dim):
    # Generate random unitary points
    u_points = np.fft.fft(np.random.normal(0,1,(n_u, dim)), axis=-1)
    u_points = u_points/np.abs(u_points)

    # Generate random unitary points
    s_points = np.random.normal(0,1,(n_s, dim))
    s_points = np.fft.fft(np.einsum('ij,i->ij', s_points, 1/np.linalg.norm(s_points, axis=-1)), axis=-1)
    return {
        'sphere': {'real': s_points.real, 'imag': s_points.imag}, 
        'unitary': {'real': u_points.real, 'imag': u_points.imag},
    }

def get_transformed_points(state, exponent):
    e = float(exponent)
    u_points = np.fft.ifft(
        (np.array(state['unitary']['real']) + 1j*np.array(state['unitary']['imag']))**e,
    axis=-1).real
    s_points = np.fft.ifft(
        (np.array(state['sphere']['real']) + 1j*np.array(state['sphere']['imag']))**e,
    axis=-1).real
    return s_points, u_points

PARAMS = {
    # min, max, default
    'dim': (1, 6, 3),
    'exp': (1, 100, 1),
}

N_POINTS = {
    # sphere, unitary
    1: (100, 20),
    2: (200, 20),
    3: (500, 200),
    4: (500, 500),
    5: (500, 500),
    6: (500, 1000),
}

app = Dash(__name__)
app.layout = html.Div([
    dcc.Store(
        data = {dim: get_random_points(*N_POINTS[dim], dim) for dim in range(PARAMS['dim'][0], PARAMS['dim'][1]+1)},
        id = 'state',
    ),
    html.H1('HRR Iterated Convolution'),
    html.Div(children=dcc.Graph(), id='graph-content'),
    html.Hr(style={'margin-bottom':'15px'}),
    "Dimension: ",
    dcc.Input(id='dimension', type='number', min=PARAMS['dim'][0], max=PARAMS['dim'][1], value=PARAMS['dim'][2], style={'margin-inline':'5px'}, n_submit=0),
    html.P('Exponent:'),
    dcc.Slider(
        id='exponent', 
        min=PARAMS['exp'][0], 
        max=PARAMS['exp'][1], 
        value=PARAMS['exp'][2], 
        updatemode='drag',
        tooltip={"placement": "bottom", "always_visible": True},
        step=1,
        marks=None,
    ),
],style={'width':'75%', 'margin':'auto'})

@app.callback(
    Output('graph-content', 'children'),
    Output('exponent', 'value'),
    Input('dimension', 'value'),
    Input('exponent', 'value'),
    State('state', 'data'),
    State('graph-content', 'children')
)
def update_graph(dim, exponent, state, graph_object):
    new_graph = False
    if ctx.triggered_id == 'dimension':
        exponent = PARAMS['exp'][0]
    else:
        new_graph = True
    s_points, u_points = get_transformed_points(state[str(dim)], exponent)
    fig = go.Figure()
    if dim < 3:
        fig.add_traces([
            go.Scatter(
                x = s_points[:,0],
                y = [0]*len(s_points) if dim == 1 else s_points[:,1],
                name = 'Sphere',
                mode = 'markers',
            ),
            go.Scatter(
                x = u_points[:,0],
                y = [0]*len(u_points) if dim == 1 else u_points[:,1],
                name = 'Unitary',
                mode = 'markers',
            ),
        ])
        fig.update_xaxes(
            range = [-3.5, 3.5],
            constrain = 'range',
        )
        fig.update_yaxes(
            range = [-3.5, 3.5],
            scaleanchor='x',
            scaleratio=1,
        )
    else:
        fig.add_traces([
            go.Scatter3d(
                x = s_points[:,0],
                y = s_points[:,1],
                z = s_points[:,2],
                name = 'Sphere',
                mode = 'markers',
                marker=dict(size=2),
            ),
            go.Scatter3d(
                x = u_points[:,0],
                y = u_points[:,1],
                z = u_points[:,2],
                name = 'Unitary',
                mode = 'markers',
                marker=dict(size=2),

            ),
        ])
        fig.update_scenes(
            aspectmode = 'cube',
            xaxis=dict(
                range = [-3.5, 3.5],
            ),
            yaxis=dict(
                range = [-3.5, 3.5],
            ),
            zaxis = dict(
                range = [-3.5, 3.5],
            ),
            camera = dict(
                eye = dict(x = 0.5, y = 0.5, z = 0.5),
                up = dict(x = 0, y = 0, z = 1),
                center = dict(x = 0, y = 0, z = 0),
            ),
        )
    fig['layout']['uirevision'] = True
    graph_object = dcc.Graph(figure=fig)
    return graph_object, exponent

if __name__ == '__main__':
    app.run(debug=False)