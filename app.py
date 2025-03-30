import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading

from train import global_training_progress, global_final_results, run_experiment_in_thread

# ============================
# Dash应用配置
# ============================
COLOR_SCHEME = {
    'primary': '#2A5C8A',    # 深海蓝
    'secondary': '#FF6F61',  # 珊瑚橙
    'accent': '#00C1D4',     # 科技蓝
    'success': '#7BC74D',     # 生态绿
    'text': '#3E4A5E',       # 深灰蓝
    'grid': '#E8ECF1',        # 浅灰
    'in_text': '#FFA500'
}

BASE_LAYOUT = dict(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font_family="Open Sans",
    font_color=COLOR_SCHEME['text'],
    hoverlabel=dict(
        bgcolor=COLOR_SCHEME['primary'],
        font_size=12,
        font_family="Open Sans",
        font_color=COLOR_SCHEME['in_text'],
    ),
    margin=dict(l=50, r=30, t=80, b=50),
    hovermode="x unified"
)

app = dash.Dash(__name__, suppress_callback_exceptions=True, assets_folder='assets')
server = app.server

# ============================
# 应用布局（保持优化版）
# ============================
app.layout = html.Div([
    html.Div([
        html.H1("🧠 AMP-GNN 动态训练监控系统", className="header-title"),
        html.P("实时追踪图神经网络训练过程的多维度指标", className="header-description")
    ], className="header"),

    html.Div([
        html.Div([
            html.Div([
                html.Label("选择数据折", className="dropdown-label"),
                dcc.Dropdown(
                    id='fold-selector',
                    options=[{'label': f'Fold {i + 1}', 'value': i} for i in range(10)],
                    value=0,
                    clearable=False,
                    className="dropdown"
                )
            ], className="dropdown-container"),

            html.Div(id='training-status', className="status-card")
        ], className="control-section"),

        dcc.Graph(id='progress-graph', className="graph-card"),

        html.Div([
            dcc.Graph(id='layer-probs-graph', className="graph-card"),
            dcc.Graph(id='similarity-graph', className="graph-card")
        ], className="grid-container")
    ], className="main-container"),

    html.Div([
        html.H2("🔬 交叉验证综合结果", className="section-title"),
        dcc.Graph(id='final-results-graph', className="full-width-graph")
    ], className="results-section"),

    dcc.Interval(id='interval-component', interval=3000, n_intervals=0),
], style={'backgroundColor': '#f5f7fa'})


# ============================
# 回调函数（适配真实数据源）
# ============================
@app.callback(
    [Output('progress-graph', 'figure'),
     Output('training-status', 'children')],
    [Input('fold-selector', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_progress(selected_fold, n):
    fold_data = global_training_progress.get(selected_fold, {})
    if fold_data and fold_data.get("epoch"):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=fold_data["epoch"],
            y=fold_data["train_loss"],
            mode='lines+markers',
            name='训练损失',
            line=dict(color=COLOR_SCHEME['secondary'], width=2),
            marker=dict(size=6, opacity=0.6)
        ))

        fig.add_trace(go.Scatter(
            x=fold_data["epoch"],
            y=fold_data["val_loss"],
            mode='lines+markers',
            name='验证损失',
            line=dict(color=COLOR_SCHEME['accent'], width=2),
            marker=dict(size=6, opacity=0.6)
        ))

        fig.update_layout(
            **BASE_LAYOUT,
            title={
                'text': f'📉 Fold {selected_fold + 1} 训练动态',
                'font': {'size': 20, 'color': COLOR_SCHEME['primary']}
            },
            xaxis=dict(
                title='训练轮次',
                gridcolor=COLOR_SCHEME['grid'],
                showline=True
            ),
            yaxis=dict(
                title='损失值',
                gridcolor=COLOR_SCHEME['grid'],
                showline=True
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        status = f"Fold {selected_fold + 1} | 当前轮次: {fold_data['epoch'][-1] + 1} | 最新训练损失: {fold_data['train_loss'][-1]:.4f}"
        return fig, status

    return go.Figure(), "等待训练数据初始化..."


@app.callback(
    Output('final-results-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_final_results(n):
    if global_final_results and global_final_results.get("folds"):
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=[f'Fold {i + 1}' for i in range(10)],
            y=global_final_results["folds"],
            marker=dict(
                color=COLOR_SCHEME['secondary'],
                line=dict(color=COLOR_SCHEME['primary'], width=1)
            ),
            opacity=0.9,
            text=[f"{acc:.2%}" for acc in global_final_results["folds"]],
            textposition='outside'
        ))

        stats_text = (f"平均准确率: {global_final_results['mean']:.2%} ± "
                      f"{global_final_results['std']:.2%}")

        fig.update_layout(
            **BASE_LAYOUT,
            title={
                'text': "📊 10折交叉验证结果分析",
                'font': {'size': 22, 'color': COLOR_SCHEME['primary']}
            },
            xaxis=dict(
                title="数据折",
                gridcolor=COLOR_SCHEME['grid'],
                showline=True
            ),
            yaxis=dict(
                title="准确率",
                tickformat=".2%",
                range=[0.6, 1.0],
                gridcolor=COLOR_SCHEME['grid'],
                showline=True
            ),
            annotations=[
                dict(
                    x=0.5, y=-0.25,
                    xref="paper", yref="paper",
                    text=stats_text,
                    showarrow=False,
                    font=dict(size=13, color=COLOR_SCHEME['accent'])
                )
            ]
        )
        return fig
    return go.Figure()


@app.callback(
    Output('layer-probs-graph', 'figure'),
    [Input('fold-selector', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_layer_probs(selected_fold, n):
    fold_data = global_training_progress.get(selected_fold, {})
    if fold_data and fold_data.get("layer_probs"):
        probs_data = fold_data["layer_probs"]
        num_layers = len(probs_data[0]['in_probs']) if probs_data else 0

        fig = make_subplots(
            rows=num_layers, cols=1,
            subplot_titles=[f"第 {i + 1} 层概率动态" for i in range(num_layers)]
        )

        for layer in range(num_layers):
            fig.add_trace(go.Scatter(
                x=fold_data["epoch"],
                y=[p['in_probs'][layer] for p in probs_data],
                name=f'输入概率 L{layer + 1}',
                line=dict(color=COLOR_SCHEME['secondary'], width=1.5),
                mode='lines+markers',
                marker=dict(size=4)
            ), row=layer + 1, col=1)

            fig.add_trace(go.Scatter(
                x=fold_data["epoch"],
                y=[p['out_probs'][layer] for p in probs_data],
                name=f'输出概率 L{layer + 1}',
                line=dict(color=COLOR_SCHEME['accent'], width=1.5),
                mode='lines+markers',
                marker=dict(size=4)
            ), row=layer + 1, col=1)

        fig.update_layout(
            **BASE_LAYOUT,
            height=300 * num_layers,
            showlegend=False
        )

        for i in range(num_layers):
            fig.update_yaxes(
                range=[0, 1],
                row=i + 1,
                col=1,
                title_text="概率值" if i == num_layers // 2 else None
            )
            fig.update_xaxes(row=i + 1, col=1, title_text="Epoch" if i == num_layers - 1 else None)

        return fig
    return go.Figure()


@app.callback(
    Output('similarity-graph', 'figure'),
    [Input('fold-selector', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_similarity_graph(selected_fold, n):
    fold_data = global_training_progress.get(selected_fold, {})
    if fold_data and fold_data.get("similarity"):
        similarity_data = fold_data["similarity"]
        num_layers = len(similarity_data[0]) if similarity_data else 0

        fig = go.Figure()

        colors = [COLOR_SCHEME['secondary'], COLOR_SCHEME['accent'], COLOR_SCHEME['success']]
        for layer in range(num_layers):
            fig.add_trace(go.Scatter(
                x=fold_data["epoch"],
                y=[d[layer] for d in similarity_data],
                name=f'Layer {layer + 1}',
                line=dict(color=colors[layer % 3], width=2),
                mode='lines+markers',
                marker=dict(size=6)
            ))

        fig.update_layout(
            **BASE_LAYOUT,
            title={
                'text': "🔍 分层特征相似度动态",
                'font': {'size': 20, 'color': COLOR_SCHEME['primary']}
            },
            xaxis=dict(
                title="训练轮次",
                gridcolor=COLOR_SCHEME['grid']
            ),
            yaxis=dict(
                title="相似度指标",
                gridcolor=COLOR_SCHEME['grid']
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        return fig
    return go.Figure()


if __name__ == '__main__':
    # 启动训练线程
    training_thread = threading.Thread(target=run_experiment_in_thread, daemon=True)
    training_thread.start()

    # 启动Dash应用
    app.run_server(debug=True, use_reloader=False)