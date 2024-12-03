from dash import Dash, html, dcc, callback, Output, Input, ctx, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from functools import partial

def rff_proj_1D(X, rand_samples):
    identity = np.ones(len(rand_samples), dtype=complex)
    proj = np.exp(1j*rand_samples)
    proj_tensor = np.einsum('i,j->ij', proj, np.ones(len(X)))
    powers = np.einsum('j,i->ij', X, identity)
    return (1/len(rand_samples))*(np.einsum('ij, i->j', np.power(proj_tensor, powers), identity)).real

def rff_proj_2D(XY, rand_samples, mean=None):
    proj = np.exp(1j*np.einsum('ij,ik->jk', XY, rand_samples))
    mean = np.zeros(2) if mean is None else mean
    center = np.exp(1j*np.einsum('i,ik->k', mean, rand_samples))
    return (1/rand_samples.shape[-1])*(np.einsum('jk, k->j', proj, center.conj())).real

PAGES = [
    'Gaussian-1D', 
    'Gaussian-2D',
    'Uniform-1D',
    'Uniform-2D',
    'Periodic-2',
    'Periodic-3',
    'Periodic-5',
    'Periodic-7',
    'Periodic-11',
]

FUNCS = {
    'Gaussian-1D': lambda x: np.exp(-0.5*x**2),
    'Gaussian-2D': lambda x: np.exp(-0.5*np.linalg.norm(x,axis=0)**2),
    'Uniform-1D': lambda x: np.nan_to_num(np.sin(np.pi*x)/(np.pi*x), posinf=0.0, neginf=0.0),
    'Uniform-2D': lambda x: np.nan_to_num((np.sin(np.pi*x[0,:])*np.sin(np.pi*x[1,:]))/((np.pi*x[0,:])*(np.pi*x[1,:])), posinf=0.0, neginf=0.0),
}

periodic_sampler = lambda n: (
    lambda r, k: np.array([
        [r*np.cos(2*j*np.pi/n + k) for j in range(n)],
        [r*np.sin(2*j*np.pi/n + k) for j in range(n)],
    ])
)

SAMPLERS = {
    'Gaussian-1D': lambda n: np.random.normal(0, 1, n),
    'Gaussian-2D': lambda n: np.random.normal(0, 1, (2, n)),
    'Uniform-1D': lambda n: np.random.uniform(-np.pi, np.pi, n),
    'Uniform-2D': lambda n: np.random.uniform(-np.pi, np.pi, (2,n)),
    'Periodic-2': lambda radius, theta: periodic_sampler(2)(radius, theta),
    'Periodic-3': lambda radius, theta: periodic_sampler(3)(radius, theta),
    'Periodic-5': lambda radius, theta: periodic_sampler(5)(radius, theta),
    'Periodic-7': lambda radius, theta: periodic_sampler(7)(radius, theta),
    'Periodic-11': lambda radius, theta: periodic_sampler(11)(radius, theta),
}

app = Dash(__name__)
app.config.suppress_callback_exceptions = True
app.layout = html.Div([
    html.H1('Random Fourier Features', style={'align':'center'}),
    dcc.Store(id='page-data', data={}),
    html.Div([
        dcc.Dropdown(
            ['1D', '2D', 'Periodic'],
            '1D',
            id='main-selector',
            clearable=False, searchable=False,
            style={'margin':'5px 0px 5px 0px'}
        ),
    ]),
    html.Div([
        dcc.Dropdown(
            id='page-selection', 
            clearable=False, searchable=False
        ),
    ]),

    html.Div(children=[], id='page-content'),
], style={'width':'75%', 'margin-inline': 'auto', 'padding':'20px', 'margin-bottom':'10%'})

@callback(
    Output('page-selection', 'options'),
    Input('main-selector', 'value'))
def set_dropdown_options(category):
    return [{'label': i, 'value': i} for i in PAGES if category in i]

@callback(
    Output('page-selection', 'value'),
    Input('page-selection', 'options'))
def set_dropdown_value(available_options):
    return available_options[0]['value']

@app.callback(
    Output('page-data', 'data'),
    Output('page-content', 'children'),
    Input('page-selection', 'value'),
    State('page-data', 'data'),
    prevent_initial_call=True,
)
def update_page(page_name, data_state):
    if '1D' in page_name:
        return page_1d(page_name, data_state)
    elif '2D' in page_name:
        return page_2d(page_name, data_state)
    elif 'Periodic' in page_name:
        return page_grid_cell(page_name, data_state)

def page_grid_cell(name, data_state, radius=1, theta=0, n_points=30, lrange=[-10, 10]):
    lrange = np.linspace(*lrange, n_points)
    if name not in data_state:
        X, Y = np.meshgrid(lrange, lrange, indexing='ij')
        data_state[name] = dict(
            samples = SAMPLERS[name](radius, theta),
            XY = np.concatenate([X.ravel()[None,:], Y.ravel()[None,:]], axis=0),
            proj_vals = None,
            lrange = lrange,
            r = radius,
            t = theta,
        )
    rand_samples = data_state[name]['samples']
    radius = data_state[name]['r']
    theta = data_state[name]['t']
    XY = data_state[name]['XY']
    proj_vals = data_state[name]['proj_vals']
    fig, proj_vals = make_figs_gridcell(XY, rand_samples, lrange, proj_vals=proj_vals)
    data_state[name]['proj_vals'] = proj_vals

    button_style = {'margin': '0px 5px 0px 5px'}
    content = html.Div([
        html.Div([
            dcc.Graph(
                figure = fig,
                id = f'{name}-ref-graph', 
            ),
        ], id=f'{name}-graph-content'),
        html.Div([
            html.P("Theta", style=button_style),
            dcc.Slider(
                min=-3.14, 
                max=3.14, 
                value=theta, 
                id=f'{name}-slider-theta',
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode='drag',
            ),
        ]),
        html.Div([
            html.P("Radius", style=button_style),
            dcc.Slider(
                min=0.05, 
                max=4, 
                value=radius, 
                step=0.05, 
                id=f'{name}-slider-radius',
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode='drag',
            ),
        ]),
        html.Button('Update', id=f'{name}-update-samples', style=button_style, n_clicks=0),
    ])
    return data_state, content

def make_figs_gridcell(XY, rand_samples, lrange, proj_vals=None):
    XY = np.array(XY)
    rand_samples = np.array(rand_samples)
    proj_vals = rff_proj_2D(XY, rand_samples) if proj_vals is None else np.array(proj_vals)

    N = len(lrange)
    lim = max(max(np.abs(rand_samples[0,:])), max(np.abs(rand_samples[1,:]))) + 0.1

    fig = make_subplots(rows=1, cols=2, subplot_titles=('RFF Samples', 'Periodic Activation'))
    fig.add_trace(
        go.Scatter(
            x = rand_samples[0,:], 
            y = rand_samples[1,:],
            mode='markers', 
            showlegend=False,
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=proj_vals.reshape(N,N), x=lrange, y=lrange),
        row=1, col=2
    )
    fig.update_layout(
        xaxis = dict(range=[-lim, lim]),
        yaxis = dict(range=[-lim, lim])
        )
    return fig, proj_vals

def update_gridcell_state(name, data_state, fig, radius, theta, *buttons):
    samples = np.array(data_state[name]['samples'])
    r = data_state[name]['r']
    t = data_state[name]['t']
    lrange = data_state[name]['lrange']
    if r == radius and t == theta:
        return data_state, fig
    data_state[name]['r'] = radius
    data_state[name]['t'] = theta
    samples = SAMPLERS[name](radius, theta)
    data_state[name]['samples'] = samples
    XY = data_state[name]['XY']
    fig, proj_vals = make_figs_gridcell(XY, samples, lrange)
    data_state[name]['proj_vals'] = proj_vals
    return data_state, fig

def page_2d(name, data_state, n_samples=100, n_points=30, lrange=[-10, 10]):
    lrange = np.linspace(*lrange, n_points)
    if name not in data_state:
        X, Y = np.meshgrid(lrange, lrange, indexing='ij')
        data_state[name] = dict(
            samples = SAMPLERS[name](n_samples),
            XY = np.concatenate([X.ravel()[None,:], Y.ravel()[None,:]], axis=0),
            proj_vals = None,
            mean = [0,0],
            lrange = lrange,
        )
    rand_samples = data_state[name]['samples']
    XY = data_state[name]['XY']
    mean = data_state[name]['mean']
    proj_vals = data_state[name]['proj_vals']
    fig, proj_vals = make_figs_2d(XY, FUNCS[name], rand_samples, lrange, proj_vals=proj_vals, mean=np.array(mean))
    data_state[name]['proj_vals'] = proj_vals

    button_style = {'margin': '0px 5px 0px 5px'}
    content = html.Div([
        html.Div([
            dcc.Graph(
                figure = fig,
                id = f'{name}-ref-graph', 
            ),
        ], id=f'{name}-graph-content'),
        html.Div([
            html.P("No. Samples:", style=button_style),
            dcc.Input(id=f'{name}-input-samples', type='number', value=len(rand_samples[0]), min=0, style=button_style, n_submit=0),
            html.Button('Update', id=f'{name}-update-samples', style=button_style, n_clicks=0),
            html.Button('Randomize', id=f'{name}-randomize-samples', style=button_style, n_clicks=0),
        ], style={'display':'flex','align-items':'center', 'justify-content':'left', 'width':'100%'}),
        html.P(),
        html.Div([
            html.P("Mean:", style=button_style),
            html.P("X", style=button_style),
            dcc.Input(id=f'{name}-input-meanx', type='number', value=mean[0], min=lrange[0], max=lrange[-1], style=button_style),
            html.P("Y", style=button_style),
            dcc.Input(id=f'{name}-input-meany', type='number', value=mean[1], min=lrange[0], max=lrange[-1], style=button_style),
            html.Button('Update', id=f'{name}-update-mean', style=button_style, n_clicks=0),
        ], style={'display':'flex','align-items':'center', 'justify-content':'left', 'width':'100%'}),
    ])
    return data_state, content

def make_figs_2d(XY, func, rand_samples, lrange, proj_vals=None, mean=np.zeros(2)):
    XY = np.array(XY)
    rand_samples = np.array(rand_samples)
    proj_vals = rff_proj_2D(XY, rand_samples, mean=mean) if proj_vals is None else np.array(proj_vals)

    N = len(lrange)
    XYmean = np.empty(XY.shape)
    for i in range(len(mean)):
        XYmean[i,:] = XY[i,:]-mean[i]

    fig = make_subplots(rows=1, cols=3, subplot_titles=('RFF Samples', 'RFF Aproximation', 'Ground Truth'))
    fig.add_trace(
        go.Scatter(x=rand_samples[0,:], y=rand_samples[1,:], mode='markers', showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=proj_vals.reshape(N,N), x=lrange, y=lrange, coloraxis='coloraxis'),
        row=1, col=2
    )
    fig.add_trace(
        go.Heatmap(z=func(XYmean).reshape(N,N), x=lrange, y=lrange, coloraxis='coloraxis'),
        row=1, col=3
    )
    fig.update_layout(coloraxis = {'colorscale':'viridis'})
    return fig, proj_vals

def update_2D_state(name, data_state, n_samples, fig, meany, meanx, *buttons):
    samples = np.array(data_state[name]['samples'])
    mean = data_state[name]['mean']
    if mean[0] is None:
        mean[0] = 0
    if mean[1] is None:
        mean[1] = 0
    lrange = data_state[name]['lrange']
    triggered = ctx.triggered_id
    if triggered in [f'{name}-update-samples', f'{name}-input-samples']:
        if n_samples > samples.shape[-1]:
            samples = np.concatenate([samples, SAMPLERS[name](n_samples-samples.shape[-1])], axis=1)
        elif n_samples < samples.shape[-1]:
            samples = samples[:,:n_samples]
        else:
            return data_state, fig
    elif triggered == f'{name}-randomize-samples':
        samples = SAMPLERS[name](n_samples)
    elif triggered in [f'{name}-update-mean', f'{name}-input-meanx', f'{name}-input-meany']:
        mean = np.array([meanx, meany])
    data_state[name]['samples'] = samples
    data_state[name]['mean'] = mean
    XY = data_state[name]['XY']
    fig, proj_vals = make_figs_2d(XY, FUNCS[name], samples, lrange, mean=mean)
    data_state[name]['proj_vals'] = proj_vals
    return data_state, fig

def page_1d(name, data_state, n_samples=10, n_points=200, lrange=[-10, 10]):
    if name not in data_state:
        data_state[name] = dict(
            samples = SAMPLERS[name](n_samples),
            X = np.linspace(*lrange, n_points),
            proj_vals = None,
        )
    rand_samples = data_state[name]['samples']
    X = data_state[name]['X']
    proj_vals = data_state[name]['proj_vals']
    fig, proj_vals = make_figs_1d(X, FUNCS[name], rand_samples, proj_vals=proj_vals)
    data_state[name]['proj_vals'] = proj_vals

    button_style = {'margin': '0px 5px 0px 5px'}
    content = html.Div([
        html.Div([
            dcc.Graph(
                figure = fig,
                id = f'{name}-ref-graph', 
            ),
        ], id=f'{name}-graph-content'),
        html.Div([
            html.P("No. Samples:", style=button_style),
            dcc.Input(id=f'{name}-input-samples', type='number', value=len(rand_samples), min=0, style=button_style, n_submit=0),
            html.Button('Update', id=f'{name}-update-samples', style=button_style, n_clicks=0),
            html.Button('Randomize', id=f'{name}-randomize-samples', style=button_style, n_clicks=0),
        ], style={'display':'flex','align-items':'center', 'justify-content':'left', 'width':'100%'})
    ])
    return data_state, content

def make_figs_1d(X, func, rand_samples, proj_vals=None):
    X = np.array(X)
    rand_samples = np.array(rand_samples)
    f1 = [
        go.Scatter(x=rand_samples, y=[0]*len(rand_samples), mode='markers', showlegend=False)
    ]
    proj_vals = rff_proj_1D(X, rand_samples) if proj_vals is None else proj_vals
    f2 = [
        go.Scatter(
            x = X, 
            y = func(X), 
            name = 'Ground Truth', 
            marker = dict(color='royalblue'), 
        ),
        go.Scatter(
            x = X, 
            y = proj_vals, 
            name = 'RFF Approximation', 
            marker = dict(color='red'),
            line = dict(dash='dot'),
        ),
    ]
    fig = make_subplots(rows=1, cols=2, subplot_titles=('RFF Samples', 'Functions'))
    for trace in f1:
        fig.add_trace(trace, row=1, col=1)
    for trace in f2:
        fig.add_trace(trace, row=1, col=2)
    return fig, proj_vals

def update_1D_state(name, data_state, n_samples, fig, *buttons):
    samples = data_state[name]['samples']
    triggered = ctx.triggered_id
    if triggered in [f'{name}-update-samples', f'{name}-input-samples']:
        if n_samples > len(samples):
            samples = np.r_[samples, SAMPLERS[name](n_samples-len(samples))]
        elif n_samples < len(samples):
            samples = samples[:n_samples]
        else:
            return data_state, fig
    elif triggered == f'{name}-randomize-samples':
        samples = SAMPLERS[name](n_samples)
    data_state[name]['samples'] = samples
    X = data_state[name]['X']
    fig, proj_vals = make_figs_1d(X, FUNCS[name], samples)
    data_state[name]['proj_vals'] = proj_vals
    return data_state, fig

PAGE_UPDATE_FUNCS = []
for name in PAGES:
    if '1D' in name:
        f = app.callback(
            Output('page-data', 'data', allow_duplicate=True),
            Output(f'{name}-ref-graph', 'figure'),
            State('page-data', 'data'),
            State(f'{name}-input-samples', 'value'),
            State(f'{name}-ref-graph', 'figure'),
            Input(f'{name}-update-samples', 'n_clicks'),
            Input(f'{name}-randomize-samples', 'n_clicks'),
            Input(f'{name}-input-samples', 'n_submit'),
            prevent_initial_call=True,
        )(partial(update_1D_state, name))
        PAGE_UPDATE_FUNCS.append(f)
    elif '2D' in name:
        f = app.callback(
            Output('page-data', 'data', allow_duplicate=True),
            Output(f'{name}-ref-graph', 'figure'),
            State('page-data', 'data'),
            State(f'{name}-input-samples', 'value'),
            State(f'{name}-ref-graph', 'figure'),
            State(f'{name}-input-meanx', 'value'),
            State(f'{name}-input-meany', 'value'),
            Input(f'{name}-update-samples', 'n_clicks'),
            Input(f'{name}-randomize-samples', 'n_clicks'),
            Input(f'{name}-update-mean', 'n_clicks'),
            Input(f'{name}-input-samples', 'n_submit'),
            Input(f'{name}-input-meanx', 'n_submit'),
            Input(f'{name}-input-meany', 'n_submit'),
            prevent_initial_call=True,
        )(partial(update_2D_state, name))
        PAGE_UPDATE_FUNCS.append(f)
    elif 'Periodic' in name:
        f = app.callback(
            Output('page-data', 'data', allow_duplicate=True),
            Output(f'{name}-ref-graph', 'figure'),
            State('page-data', 'data'),
            State(f'{name}-ref-graph', 'figure'),
            Input(f'{name}-slider-radius', 'value'),
            Input(f'{name}-slider-theta', 'value'),
            Input(f'{name}-update-samples', 'n_clicks'),
            prevent_initial_call=True,
        )(partial(update_gridcell_state, name))
        PAGE_UPDATE_FUNCS.append(f)

if __name__ == '__main__':
    app.run_server(debug=False)