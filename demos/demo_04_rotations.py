import numpy as np
from scipy.special import softmax
import plotly.graph_objects as go
from dash import Dash, html, Input, Output, ctx, State, dcc

### Main Functions ###

def rand_unit_vec(n):
    x = np.random.normal(0,1,n)
    return x/np.linalg.norm(x)

def rand_unitary(dim):
    m = np.random.normal(0,1,(dim, dim)) + 1j*np.random.normal(0,1,(dim,dim))
    q, r = np.linalg.qr(m)
    scale = np.diagonal(r)
    scale = scale/np.abs(scale)
    return q@np.diag(scale)

def rand_special_unitary(dim):
    m = np.random.normal(0,1,(dim, dim)) + 1j*np.random.normal(0,1,(dim,dim))
    q, r = np.linalg.qr(m)
    scale = np.diagonal(r)
    scale = scale/np.abs(scale)
    q = q@np.diag(scale)
    eigvals, eigvecs = np.linalg.eig(q)
    eigvals[-1] = np.prod(eigvals[:-1]).conj()
    return eigvecs@np.diag(eigvals)@eigvecs.T.conj()

def mdot(a:np.array,b:np.array):
    assert a.shape == b.shape and a.shape[-1] == a.shape[-2]
    return (1/a.shape[-1])*np.trace(a.T.conj()@b).real

def similarity(a, b):
    return (1/a.shape[-1])*np.sum(a.real*b.real + a.imag*b.imag)

def get_random_memory(n_items):
    mem_matrix = np.array([np.eye(2,2, dtype=complex)]+[rand_special_unitary(2) for i in range(n_items-1)])
    mem_flat = np.array([np.r_[mem_matrix[i,:,:].real.ravel(), mem_matrix[i,:,:].imag.ravel()] for i in range(n_items)])
    eigvals, eigvecs = np.linalg.eig(mem_matrix)
    return mem_matrix, mem_flat, eigvals, eigvecs

def retrieve(x, mem, eigvecs, eigvals, beta=1):
    # works on flattened x, mem
    assert x.shape[-1] == mem.shape[-1]
    activation = softmax((beta/eigvals.shape[-1])*mem@x)
    powers = np.outer(activation, np.ones(eigvals.shape[-1]))
    eigvals = np.power(eigvals, powers)
    mem = np.einsum('ijk, ik, imk->ijm', eigvecs, eigvals, eigvecs.conj())
    echo = np.linalg.multi_dot(mem)
    return echo

def resonate(state, max_steps = 50, tol = 1e-6, conj = True, norm_output = False, final_path_steps = 10):
    # initialize source, target
    x = np.array(state['source'])
    xq = vec3_to_quat(x)
    y = np.array(state['target'])
    yq = vec3_to_quat(y)

    # initialize guesses
    a = rand_special_unitary(2)
    b = rand_special_unitary(2)

    # params
    beta = float(state['params']['beta'])
    d1 = mdot(xq,yq)
    d2 = 0.0

    # memory
    eigvals = np.array(state['memory']['eigvals']['real']) + 1j*np.array(state['memory']['eigvals']['imag'])
    eigvecs = np.array(state['memory']['eigvecs']['real']) + 1j*np.array(state['memory']['eigvecs']['imag'])
    mem_flat = np.array(state['memory']['flat']['real'])
    
    # data collection
    update_path = dict(x=[x[0]], y=[x[1]], z=[x[2]])

    # resonate
    for _ in range(max_steps):
        if conj:
            probe = yq@(xq@b.conj().T).conj().T
            probe_flat = np.r_[probe.real.ravel(), probe.imag.ravel()]
            echo = retrieve(probe_flat, mem_flat, eigvecs, eigvals, beta = beta)
            a = echo
            probe = ((a@xq).conj().T@yq).conj().T
            probe_flat = np.r_[probe.real.ravel(), probe.imag.ravel()]
            echo = retrieve(probe_flat, mem_flat, eigvecs, eigvals, beta = beta)
            b = echo
            res = a@xq@b.conj().T
        else:
            probe = yq@(xq@b).conj().T
            probe_flat = np.r_[probe.real.ravel(), probe.imag.ravel()]
            echo = retrieve(probe_flat, mem_flat, eigvecs, eigvals, beta = beta)
            a = echo
            probe = (a@xq).conj().T@yq
            probe_flat = np.r_[probe.real.ravel(), probe.imag.ravel()]
            echo = retrieve(probe_flat, mem_flat, eigvecs, eigvals, beta = beta)
            b = echo
            res = a@xq@b
        # early exit condition
        d2 = mdot(res, yq)
        if np.abs(d1-d2) < tol:
            break
        d1 = d2
        xf = quat_to_vec3(res)
        if norm_output:
            xf = xf/np.linalg.norm(xf)
        update_path['x'].append(xf[0])
        update_path['y'].append(xf[1])
        update_path['z'].append(xf[2])
    
    res = xf
    # rotation path data
    rotation_path = dict(x=[x[0]], y=[x[1]], z=[x[2]])
    eigvalsa, eigvecsa = np.linalg.eig(a)
    eigvalsb, eigvecsb = np.linalg.eig(b)
    for _, i in enumerate(np.linspace(0, 1, final_path_steps)):
        if conj:
            xf = (eigvecsa @ np.diag(eigvalsa**i) @ (eigvecsa).conj().T) @ xq @ (eigvecsb @ np.diag(eigvalsb**i) @ (eigvecsb).conj().T).conj().T
        else:
            xf = (eigvecsa @ np.diag(eigvalsa**i) @ (eigvecsa).conj().T) @ xq @ (eigvecsb @ np.diag(eigvalsb**i) @ (eigvecsb).conj().T)
        xf = quat_to_vec3(xf)
        if norm_output:
            xf = xf/np.linalg.norm(xf)
        rotation_path['x'].append(xf[0])
        rotation_path['y'].append(xf[1])
        rotation_path['z'].append(xf[2])
    return x, y, res, update_path, rotation_path

get_random_formatted_memory = lambda n: {
    key: {'real': val.real, 'imag': val.imag} for key, val in zip(['matrix', 'flat', 'eigvals', 'eigvecs'], get_random_memory(n))
}

to_quat = lambda x: np.array([
    [ x[0]+1j*x[1], x[2]+1j*x[3]],
    [-x[2]+1j*x[3], x[0]-1j*x[1]],
])
vec3_to_quat = lambda x: to_quat(np.r_[0, x])
quat_to_vec4 = lambda x: np.array([
    x[0][0].real, x[0][0].imag, x[0][1].real, x[0][1].imag
])
quat_to_vec3 = lambda x: np.array([
    x[0][0].imag, x[0][1].real, x[0][1].imag
])

### Web App ###

PARAMS = dict(
    # min, max, default
    beta = (0, 20, 6),
    mem_items = (1, 100, 10),
)
INITIAL_STATE = {
    'scene': dict(
        xaxis_visible = False, 
        yaxis_visible = False,
        zaxis_visible = False,
        camera = dict(
            eye = dict(x = 0.75, y = 0.75, z = 0.75),
            up = dict(x = 0, y = 0, z = 1),
            center = dict(x = 0, y = 0, z = 0),
        ),
    ), 
    'memory': get_random_formatted_memory(PARAMS['mem_items'][-1]), 
    'source': rand_unit_vec(3), 
    'target': rand_unit_vec(3), 
    'params': {key: val[-1] for key, val in PARAMS.items()},
}

def default_fig():
    fig = go.Figure()
    # Plot sphere for visualization
    thetas = np.linspace(0,2*np.pi, 30)
    phis = np.linspace(0,np.pi, 30)

    for phi in phis:
        fig.add_trace(
            go.Scatter3d(
                x = np.cos(thetas)*np.sin(phi),
                y = np.sin(thetas)*np.sin(phi),
                z = [np.cos(phi)]*len(thetas),
                mode = 'lines',
                line = dict(color='rgba(0,0,0,0.05)'),
                showlegend=False,
            )
        )
    for theta in thetas[1:]:
        fig.add_trace(
            go.Scatter3d(
                x = np.cos(theta)*np.sin(phis),
                y = np.sin(theta)*np.sin(phis),
                z = np.cos(phis),
                mode = 'lines',
                line = dict(color='rgba(0,0,0,0.1)'),
                showlegend=False,
            )
        )
    fig.update_scenes(**INITIAL_STATE['scene'])
    fig['layout']['uirevision'] = True
    fig.update_layout(
        legend= {'itemsizing': 'constant'}
    )
    return fig

DEFAULT_FIG = default_fig()

def update_graph(graph_object, x, y, res, update_path, rotation_path):
    fig = go.Figure(
        data = DEFAULT_FIG.data,
        layout = graph_object['props']['figure']['layout']
    )
    fig.add_traces([
        # plot target
        go.Scatter3d(
            x = [y[0]], 
            y = [y[1]], 
            z = [y[2]],  
            name ='Target', 
            mode = 'markers',
            marker = dict(symbol='x', color='red'),
        ),
        # plot x initial
        go.Scatter3d(
            x = [x[0]], 
            y = [x[1]], 
            z = [x[2]], 
            name ='Initial Position', 
            mode = 'markers',
            marker = dict(symbol='circle', color='blue')
        ),
        # plot x final
        go.Scatter3d(
            x = [res[0]], 
            y = [res[1]], 
            z = [res[2]], 
            name = 'Final Position', 
            mode = 'markers',
            marker = dict(symbol='diamond', color='purple')
        ),
         # plot x update steps
        go.Scatter3d(
            x = update_path['x'], 
            y = update_path['y'], 
            z = update_path['z'], 
            name = 'Major Updates Path', 
            mode='lines+markers', 
            line = dict(dash='dashdot'),
            marker = dict(symbol='square-open', color='green', size=4),
        ),
        # plot x rotation steps
        go.Scatter3d(
            x = rotation_path['x'], 
            y = rotation_path['y'], 
            z = rotation_path['z'], 
            name = 'Rotation Path',
            mode='lines', 
            line = dict(dash='dashdot', width=4, color='blue'),
        ),
    ])
    graph_object['props']['figure']['data'] = fig.data
    graph_object['props']['figure']['layout'] = fig.layout
    return graph_object

button_style = {'margin': '0px 5px 0px 5px'}
input_container_style = {'margin-top': '5px', 'margin-bottom':'5px', 'padding-left':'50px'}
control_container_style = {'display':'flex', 'justify-content':'center'}

app = Dash(__name__)
app.layout = html.Div([
    dcc.Store(
        data = INITIAL_STATE,
        id = 'state',
    ),
    html.H1('Unitary Group VSA'),
    html.H3('SU(2) Rotation Model'),
    html.Div(
        children = dcc.Graph(figure=default_fig()),
        id = 'graph-content'
    ),
    html.Hr(),
    html.Div([        
        html.Div([
            html.H4('Randomize'),
            html.Button('Source', id='source-button', disabled=False, style=button_style),
            html.Button('Target', id='target-button', disabled=False, style=button_style),
            html.Button('Memory', id='memory-button', disabled=False, style=button_style),
        ]),
        html.Div([
            html.Div([
                html.H4('Parameters'),
                'Beta: ',
                dcc.Input(
                    id = 'param-beta',
                    type = 'number', 
                    min = PARAMS['beta'][0], 
                    max = PARAMS['beta'][1], 
                    value = PARAMS['beta'][2], 
                ),
                html.Button('↻', id='beta-button', style=button_style, disabled=False)
            ], style = input_container_style),
            html.Div([
                'Mem. Items: ',
                dcc.Input(
                    id = 'param-memory',
                    type = 'number', 
                    min = PARAMS['mem_items'][0], 
                    max = PARAMS['mem_items'][1], 
                    value = PARAMS['mem_items'][2], 
                ),
            ], style = input_container_style),
            html.Div([
                dcc.Checklist(
                    ['Conj', 'Norm'],
                    ['Conj', 'Norm'],
                    id = 'param-algoptions',
                    inline = True,
                ),
            ], style = input_container_style),

        ]),
    ], style = control_container_style),
    html.P(id='some-text', children=[]),
    html.Div(id='trigger', children=None, style={'display':'none'})
], style={'width':'75%', 'margin':'auto'})

@app.callback(
    Output('trigger', 'children'),
    Output('source-button', 'disabled', allow_duplicate=True),
    Output('target-button', 'disabled', allow_duplicate=True),
    Output('memory-button', 'disabled', allow_duplicate=True),
    Output('beta-button', 'disabled', allow_duplicate=True),
    Input('source-button', 'n_clicks'),
    Input('target-button', 'n_clicks'),
    Input('memory-button', 'n_clicks'),
    Input('beta-button', 'n_clicks'),
    prevent_initial_call=True,
)
def disable_buttons(*buttons):
    triggered = ctx.triggered_id
    return triggered, *tuple([True]*4)
    
@app.callback(
    Output('state', 'data'),
    Output('graph-content', 'children'),
    Output('source-button', 'disabled'),
    Output('target-button', 'disabled'),
    Output('memory-button', 'disabled'),
    Output('beta-button', 'disabled'),
    Input('trigger', 'children'),
    State('state', 'data'),
    State('graph-content', 'children'),
    State('param-beta', 'value'),
    State('param-memory', 'value'),
    State('param-algoptions', 'value'),
)
def do_action(trigger, state, graph_object, beta, n_mem, algoptions):
    if trigger == 'source-button':
        state['source'] = rand_unit_vec(3)
    elif trigger == 'target-button':
        state['target'] = rand_unit_vec(3)
    elif trigger == 'memory-button':
        state['memory'] = get_random_formatted_memory(n_mem)
    state['params']['beta'] = beta
    norm = True if 'Norm' in algoptions else False
    conj = True if 'Conj' in algoptions else False
    x, y, res, update_path, rotation_path = resonate(state, conj=conj, norm_output=norm)
    graph_object = update_graph(graph_object, x, y, res, update_path, rotation_path)
    return state, graph_object, *tuple([False]*4)
        
if __name__ == '__main__':
    app.run(debug=False)