from dash import dcc, html
import dash_bootstrap_components as dbc
import dash
import plotly.graph_objects as go
import numpy as np
from utils import Path

def create_figure(graph, paths, avg_charge, courier_speed, title, show_avg_charge=False, width=1600, height=900):
    fig = go.Figure()

    colors = np.linspace(0, 1, len(paths))

    # Add routes to the figure
    for path, color in zip(paths, colors):
        x = []
        y = []
        for vertex_id in path.indx:
            vertex = graph.vertices.get(vertex_id)
            if vertex:
                x.append(vertex.x)
                y.append(vertex.y)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=f'Route ({path.dist:.2f} m)',
                                 line=dict(color=f'rgba({color*255}, {100}, {150}, 0.8)')))

    # Add vertices to the figure
    x_stations = [v.x for v in graph.vertices.values() if v.type == 'station']
    y_stations = [v.y for v in graph.vertices.values() if v.type == 'station']
    x_scooters = [v.x for v in graph.vertices.values() if v.type == 'scooter']
    y_scooters = [v.y for v in graph.vertices.values() if v.type == 'scooter']

    fig.add_trace(go.Scatter(x=x_stations, y=y_stations, mode='markers', name='Stations', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x_scooters, y=y_scooters, mode='markers', name='Scooters', marker=dict(color='green')))

    # Add charge annotations
    for scooter in graph.vertices.values():
        if scooter.type == 'scooter':
            fig.add_annotation(x=scooter.x, y=scooter.y, text=f"{scooter.charge:.1f}%", showarrow=False)

    total_distance = sum(path.dist for path in paths)
    total_time_seconds = total_distance / courier_speed

    # Annotations to be added to the bottom of the graph
    font_settings = dict(size=12, color='black') 

    annotations = [
        dict(
            x=0.5,
            y=-0.12,
            xref='paper',
            yref='paper',
            text=f"Total Distance: {total_distance:.2f} m",
            showarrow=False,
            align='center',
            font=font_settings
        ),
        dict(
            x=0.5,
            y=-0.18,
            xref='paper',
            yref='paper',
            text=f"Total Time: {total_time_seconds:.2f} seconds",
            showarrow=False,
            align='center',
            font=font_settings
        ),
    ]

    if show_avg_charge:
        annotations.append(
            dict(
                x=0.5,
                y=-0.24,
                xref='paper',
                yref='paper',
                text=f"Average Charge: {avg_charge:.1f}%",
                showarrow=False,
                align='center',
                font=font_settings
            )
        )

    fig.update_layout(
        title=title,
        width=width,
        height=height,
        legend=dict(
            x=1,
            y=0.5,
            traceorder="normal",
            font=dict(size=10)
        ),
        margin=dict(t=50, b=200),  
        annotations=annotations
    )

    return fig

def create_dash_app(graph, paths, avg_charge_before, avg_charge_after, courier_speed):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    fig_before = create_figure(graph, paths, avg_charge_before, courier_speed, title="Scooter Charging Routes Before Charge", show_avg_charge=True, width=1600, height=900)
    fig_after = create_figure(graph, paths, avg_charge_after, courier_speed, title="Scooter Charging Routes After Charge", show_avg_charge=True, width=1600, height=900)

    app.layout = html.Div([
        html.H1("SmartScooter Charge & Route Optimizer", style={'textAlign': 'center'}),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_before), width=12),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_after), width=12),
        ])
    ])

    return app
