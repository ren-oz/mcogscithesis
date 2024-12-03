from dash import Dash, html, dcc, Output, Input, ctx, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.special import softmax

DATA = dict(probes=None, mem=None)
N_MEM = 5
N_PROBES = 50
betas = np.arange(0, 21, 1)
n_iters = 20

def generate_data(n_mem, n_probes):
    global DATA
    mem_angles = np.random.uniform(-np.pi, np.pi, (n_mem, 2))
    DATA['mem'] = mem_angles.copy()

    mem = np.exp(1j*mem_angles)
    mem_tensor = np.einsum('ij,k->ijk', mem, np.ones(n_probes))

    probe_angles = np.random.uniform(-np.pi, np.pi, (n_probes, 2))
    probes = np.exp(1j*probe_angles)

    data = {k:np.empty((len(betas), n_iters+1, n_probes, 2)) for k in ['product', 'polar']}

    for beta_idx, beta in enumerate(betas):
        data['product'][beta_idx, 0,:,:] = probe_angles
        data['polar'][beta_idx, 0,:,:] = probe_angles
        polar_probes = probes
        product_probes = probes
        for iter in range(1, n_iters+1):
            polar_probes = (mem.T@softmax(beta*0.5*(mem@(polar_probes.conj().T)).real, axis = 0)).T
            polar_probes = polar_probes * 1/np.abs(polar_probes)
            data['polar'][beta_idx, iter,:,:] = np.angle(polar_probes)

            product_weights = softmax(beta*0.5*np.einsum('ij, kj->ik', mem, product_probes.conj()).real, axis = 0)
            powers = np.einsum('ik, j->ijk', product_weights, np.ones(2))
            product_probes = np.prod(np.power(mem_tensor, powers), axis=0).T
            data['product'][beta_idx, iter,:,:] = np.angle(product_probes)
    DATA['probes'] = data.copy()

app = Dash(__name__)

app.layout = html.Div([
    html.H1('Associative Memory Dynamics', style={'align':'center'}),
    html.H4('2D Complex Unitary HRRs'),
    dcc.Dropdown(['Flat Torus', '3D Projection'], 'Flat Torus', id='projection-dropdown', clearable=False, searchable=False),
    html.Div(children=[], id='graph-content'),
    html.Div([
            html.Button('Randomize', id='random-button', n_clicks=0),
            html.P('No. Traces:', style={'padding':'0px 10px 0px 10px'}),
            dcc.Input(id='input-traces', type='number', value=N_MEM, min=1, max=100, n_submit=0),
            html.P('No. Probes:', style={'padding':'0px 10px 0px 10px'}),
            dcc.Input(id='input-probes', type='number', value=N_PROBES, min=1, max=100, n_submit=0),
    ], style={'display':'flex', 'justify-content':'center'}),
    html.P("Retrieval Iteration:"),
    html.Div([
        html.Button('<<', id='iter-jump-left-button', n_clicks=0, style={'margin':'0px 5px 0px 5px'}),
        html.Button('<', id='iter-left-button', n_clicks=0),
        html.Div([
            dcc.Slider(
                id='iteration-slider',
                min=0, max=n_iters, step=1,
                updatemode='drag',
                value=0,
            )
        ], style={'width':'100%'}),
        html.Button('>', id='iter-right-button', n_clicks=0),
        html.Button('>>', id='iter-jump-right-button', n_clicks=0, style={'margin':'0px 5px 0px 5px'}),
    ], style={'display':'flex', 'justify-content':'center',}),
    html.P("Beta:"),
    html.Div([
        html.Button('<<', id='beta-jump-left-button', n_clicks=0, style={'margin':'0px 5px 0px 5px'}),
        html.Button('<', id='beta-left-button', n_clicks=0),
        html.Div([
            dcc.Slider(id='beta-slider',
                min=min(betas), max=max(betas), step=abs(betas[1]-betas[0]),
                updatemode='drag',
                value=betas[len(betas)//2]),
        ], style={'width':'100%'}),
        html.Button('>', id='beta-right-button', n_clicks=0),
        html.Button('>>', id='beta-jump-right-button', n_clicks=0, style={'margin':'0px 5px 0px 5px'}),
    ], style={'display':'flex', 'justify-content':'center',}),
    dcc.Store(id='camera', data={})
], style={'width':'75%', 'margin-inline': 'auto', 'padding':'20px', 'margin-bottom':'10%'})

thetas = np.linspace(-np.pi, np.pi, 32)
xx, yy = np.meshgrid(thetas, thetas)

wireframe_torus = []
linestyle = dict(mode='lines', line=dict(color='rgba(0,0,0,0.1)', width=2), showlegend=False)
for x,y in zip(xx, yy):
    wireframe_torus += [
        go.Scatter3d(
            x = (2+np.cos(x))*np.cos(y), 
            y = (2+np.cos(x))*np.sin(y), 
            z = np.sin(x), 
            **linestyle
        ),
        go.Scatter3d(
            x = (2+np.cos(y))*np.cos(x), 
            y = (2+np.cos(y))*np.sin(x), 
            z = np.sin(y),  
            **linestyle
        ),
        ]

@app.callback(
    Output("graph-content", "children"),
    Output("iteration-slider", "value"),
    Output("beta-slider", "value"),
    Output('camera', 'data'),
    Input('camera', 'data'),
    State('graph-content', 'children'),
    Input("iteration-slider", "value"),
    Input("beta-slider", "value"),
    State('input-probes', 'value'),
    State('input-traces', 'value'),
    Input('projection-dropdown', 'value'),
    Input('random-button', 'n_clicks'),
    Input('iter-left-button', 'n_clicks'),
    Input('iter-right-button', 'n_clicks'),
    Input('iter-jump-left-button', 'n_clicks'),
    Input('iter-jump-right-button', 'n_clicks'),
    Input('beta-left-button', 'n_clicks'),
    Input('beta-right-button', 'n_clicks'),
    Input('beta-jump-left-button', 'n_clicks'),
    Input('beta-jump-right-button', 'n_clicks'),
    Input('input-probes', 'n_submit'),
    Input('input-traces', 'n_submit'),
    )
def update_bar_chart(camera, children, iter, beta, n_probes, n_traces, proj, *args):
    global DATA
    beta_idx = np.where(betas==beta)[0][0]
    change_proj = False
    match ctx.triggered_id:
        case 'projection_dropdown':
            change_proj = True
        case 'random-button' | 'input-probes' | 'input-traces':
            generate_data(n_traces, n_probes)
            iter = 0
        case 'iter-left-button':
            iter = max(0, iter-1)
        case 'iter-right-button':
            iter = min(iter+1, n_iters)
        case 'iter-jump-left-button':
            iter = 0
        case 'iter-jump-right-button':
            iter = n_iters
        case 'beta-left-button':
            beta_idx = max(0, beta_idx-1)
        case 'beta-right-button':
            beta_idx = min(beta_idx+1, len(betas)-1)
        case 'beta-jump-left-button':
            beta_idx = 0
        case 'beta-jump-right-button':
            beta_idx = len(betas)-1
    if not change_proj:
        try:
            relayout = children[0]['props']['relayoutData']
            if 'scene.camera' or 'scene2.camera' in relayout:
                camera.update(relayout)
        except Exception as e:
            relayout = None
            pass
    match proj:
        case 'Flat Torus':
            fig = make_subplots(rows=1, cols=2, subplot_titles = ('Polar', 'Product'),
                                specs=[[{}, {}]])
            for col in range(1, 3):
                fig.add_trace(
                    go.Scatter(
                        x = DATA['mem'][:,0],
                        y = DATA['mem'][:,1],
                        mode = 'markers',
                        marker = dict(
                            symbol = 'x',
                            size = 20,
                            color = 'royalblue',
                        ),
                        showlegend = bool(col%2),
                        name = 'Memory Traces'
                    ), row = 1, col = col,
                )
                fig.add_trace(
                    go.Scatter(
                        x = DATA['probes'][['polar', 'product'][(col-1)%2]][beta_idx, iter, :, 0],
                        y = DATA['probes'][['polar', 'product'][(col-1)%2]][beta_idx, iter, :, 1],
                        mode = 'markers',
                        marker = dict(
                            size = 10,
                            color = 'red',
                        ),
                        showlegend = bool(col%2),
                        name = 'Probes',

                    ), row = 1, col = col,
                )
                fig.update_xaxes(
                    range = [-3.5, 3.5],
                    constrain = 'domain',
                    row = 1, col = col,
                )
                fig.update_yaxes(
                    range = [-3.5, 3.5],
                    row = 1, col = col,
                    scaleanchor=('x' if col%2 else 'x2'),
                    scaleratio=1,
                )
                fig.add_shape(type="rect",
                    x0 = -np.pi, y0 = -np.pi, x1 = np.pi, y1 = np.pi,
                    line = dict(dash='dot', color='royalblue'),
                    row = 1, col = col,
                )
            graph = dcc.Graph(
                figure = fig,
            )
            
        case '3D Projection':
            fig = make_subplots(rows=1, cols=2, subplot_titles = ('Polar', 'Product'), 
                                specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]])
            for col in range(1, 3):
                fig.add_traces(wireframe_torus, rows = 1, cols=col)
                fig.add_trace(
                    go.Scatter3d(
                        x = (2+np.cos(DATA['mem'][:,0]))*np.cos(DATA['mem'][:,1]),
                        y = (2+np.cos(DATA['mem'][:,0]))*np.sin(DATA['mem'][:,1]),
                        z = np.sin(DATA['mem'][:,0]),
                        mode = 'markers',
                        marker = dict(
                            symbol = 'x',
                            size = 5,
                            color = 'royalblue',
                        ),
                        showlegend = bool(col%2),
                        name = 'Memory Traces'
                    ), row = 1, col = col,
                )
                fig.add_trace(
                    go.Scatter3d(
                        x = (2  + np.cos(DATA['probes'][['polar', 'product'][(col-1)%2]][beta_idx, iter, :, 0]))\
                                * np.cos(DATA['probes'][['polar', 'product'][(col-1)%2]][beta_idx, iter, :, 1]),
                        y = (2  + np.cos(DATA['probes'][['polar', 'product'][(col-1)%2]][beta_idx, iter, :, 0]))\
                                * np.sin(DATA['probes'][['polar', 'product'][(col-1)%2]][beta_idx, iter, :, 1]),
                        z = np.sin(DATA['probes'][['polar', 'product'][(col-1)%2]][beta_idx, iter, :, 0]),
                        mode = 'markers',
                        marker = dict(
                            size = 5,
                            color = 'red',
                        ),
                        showlegend = bool(col%2),
                        name = 'Probes'
                    ), row = 1, col = col,
                )

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
                xaxis_visible = False, 
                yaxis_visible = False,
                zaxis_visible = False,
                camera = dict(
                    eye = dict(x = 0.6, y = 0.6, z = 0.6),
                    up = dict(x = 0, y = 0, z = 1),
                    center = dict(x = 0, y = 0, z = 0),
                ),
            )
            if relayout is not None:
                if 'scene.camera' in relayout:
                    fig.layout.scene.camera = camera['scene.camera']
                    fig.layout.scene2.camera = camera['scene.camera']
                elif 'scene2.camera' in relayout:
                    fig.layout.scene.camera = camera['scene2.camera']
                    fig.layout.scene2.camera = camera['scene2.camera']
            else:
                if 'scene.camera' in camera:
                    fig.layout.scene.camera = camera['scene.camera']
                    fig.layout.scene2.camera = camera['scene.camera']
                elif 'scene2.camera' in camera:
                    fig.layout.scene.camera = camera['scene2.camera']
                    fig.layout.scene2.camera = camera['scene2.camera']

            graph = dcc.Graph(
                figure = fig,
            )
    return [graph], iter, betas[beta_idx], camera

if __name__ == "__main__":
    generate_data(N_MEM, N_PROBES)
    app.run_server(debug=False)