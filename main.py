from dash import dcc, html
import dash_bootstrap_components as dbc
import dash
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

Base = declarative_base()

class Vertex:
    def __init__(self, id, x, y, charge=100, type='station', battery_count=0):
        self.id = id
        self.x = x
        self.y = y
        self.charge = charge
        self.type = type
        self.battery_count = battery_count
        self.last_update = datetime.now()

    def update_charge(self, delta_time):
        if self.type == 'scooter':
            self.charge = max(0, self.charge)
        self.last_update = datetime.now()


class Graph:
    def __init__(self):
        self.vertices = {}
        self.edges = {}

    def add_vertex(self, vertex: Vertex):
        self.vertices[vertex.id] = vertex

    def add_edge(self, from_id, to_id, weight):
        if from_id not in self.edges:
            self.edges[from_id] = []
        self.edges[from_id].append((to_id, weight))
        if to_id not in self.edges:
            self.edges[to_id] = []
        self.edges[to_id].append((from_id, weight))

    def get_neighbors(self, vertex_id):
        return self.edges.get(vertex_id, [])

    def get_edge_weight(self, from_id, to_id):
        for v, w in self.edges.get(from_id, []):
            if v == to_id:
                return w
        return float('inf')

def euclidean_distance(v1: Vertex, v2: Vertex):
    return ((v1.x - v2.x)**2 + (v1.y - v2.y)**2)**0.5

class Location(Base):
    __tablename__ = 'locations'
    id = Column(String, primary_key=True)
    x = Column(Float)
    y = Column(Float)
    charge = Column(Float, default=100)
    type = Column(String)
    battery_count = Column(Integer, default=0)

class Edge(Base):
    __tablename__ = 'edges'
    id = Column(Integer, primary_key=True, autoincrement=True)
    from_id = Column(String, ForeignKey('locations.id'))
    to_id = Column(String, ForeignKey('locations.id'))
    weight = Column(Float)

def populate_database(num_stations, num_scooters, battery_count_per_station):
    Session = sessionmaker(bind=engine)
    session = Session()

    session.query(Location).delete()
    session.query(Edge).delete()
    session.commit()

    existing_ids = set()

    stations = []
    for i in range(num_stations):
        station_id = f's{i}'
        while station_id in existing_ids:
            station_id = f's{i}_{np.random.randint(0, 1000)}'
        existing_ids.add(station_id)
        stations.append(Location(id=station_id, x=np.random.uniform(0, 100), y=np.random.uniform(0, 100), type='station', battery_count=battery_count_per_station))

    scooters = []
    for i in range(num_scooters):
        scooter_id = f'sc{i}'
        while scooter_id in existing_ids:
            scooter_id = f'sc{i}_{np.random.randint(0, 1000)}'
        existing_ids.add(scooter_id)
        scooters.append(Location(id=scooter_id, x=np.random.uniform(0, 100), y=np.random.uniform(0, 100), charge=np.random.uniform(10, 80), type='scooter'))

    session.add_all(stations)
    session.add_all(scooters)

    all_locations = stations + scooters
    edges = []
    for i in range(len(all_locations)):
        for j in range(i + 1, len(all_locations)):
            loc1 = all_locations[i]
            loc2 = all_locations[j]
            distance = euclidean_distance(Vertex(loc1.id, loc1.x, loc1.y), Vertex(loc2.id, loc2.x, loc2.y))
            edges.append(Edge(from_id=loc1.id, to_id=loc2.id, weight=distance))
            edges.append(Edge(from_id=loc2.id, to_id=loc1.id, weight=distance))

    session.add_all(edges)
    session.commit()
    session.close()

def get_graph_from_db():
    Session = sessionmaker(bind=engine)
    session = Session()
    
    locations = session.query(Location).all()
    edges = session.query(Edge).all()
    
    graph = Graph()
    for loc in locations:
        vertex = Vertex(loc.id, loc.x, loc.y, loc.charge, loc.type, loc.battery_count)
        graph.add_vertex(vertex)
    
    for edge in edges:
        graph.add_edge(edge.from_id, edge.to_id, edge.weight)
    
    session.close()
    return graph

def update_graph_state(graph):
    current_time = datetime.now()
    Session = sessionmaker(bind=engine)
    session = Session()

    for vertex in graph.vertices.values():
        delta_time = current_time - vertex.last_update
        vertex.update_charge(delta_time)

        if vertex.type == 'scooter':
            location = session.query(Location).filter_by(id=vertex.id).first()
            if location:
                location.charge = vertex.charge

    session.commit()
    session.close()

def update_battery_and_charge_scooters(graph, stations, num_batteries, charged_scooters):
    Session = sessionmaker(bind=engine)
    session = Session()

    for station_id in stations:
        station = graph.vertices[station_id]
        if station.type == 'station':
            neighbors = graph.get_neighbors(station_id)
            scooters_to_charge = list(set([v for v, _ in neighbors if v.startswith('sc') and v not in charged_scooters]))  # Убираем дублирование и уже заряженные самокаты
            logging.info(f"Station {station_id} can charge up to {num_batteries} scooters. Found scooters to charge: {scooters_to_charge}")
            for scooter_id in scooters_to_charge[:num_batteries]:
                scooter = graph.vertices[scooter_id]
                if scooter.charge < 100:
                    scooter.charge = 100
                    station.battery_count -= 1
                    if station.battery_count < 0:
                        station.battery_count = 0

                    location = session.query(Location).filter_by(id=scooter_id).first()
                    if location:
                        location.charge = 100
                        logging.info(f"Charged scooter {scooter_id} to 100%")
                    station_location = session.query(Location).filter_by(id=station_id).first()
                    if station_location:
                        station_location.battery_count = station.battery_count
                        logging.info(f"Updated battery count for station {station_id}: {station.battery_count}")

                    charged_scooters.add(scooter_id)  # Добавляем в список заряженных самокатов

    session.commit()
    session.close()

def total_route_distance(route, graph):
    total_distance = 0
    for i in range(len(route) - 1):
        from_id = route[i]
        to_id = route[i + 1]
        total_distance += graph.get_edge_weight(from_id, to_id)
    return total_distance

def nearest_neighbor_route_limited(graph, start_id, limit, unvisited_scooters):
    route = [start_id]
    current_vertex = start_id

    while unvisited_scooters and len(route) - 1 < limit:
        next_vertex = min(unvisited_scooters, key=lambda vertex: graph.get_edge_weight(current_vertex, vertex))
        unvisited_scooters.remove(next_vertex)
        route.append(next_vertex)
        current_vertex = next_vertex

    return route

def build_stages(graph, start_station, num_batteries):
    unvisited_scooters = {v.id for v in graph.vertices.values() if v.type == 'scooter'}
    visited_stations = set()
    stages = []
    charged_scooters = set()  # Добавляем множество для отслеживания заряженных самокатов

    current_station = start_station
    visited_stations.add(current_station)

    avg_charge_before = np.mean([v.charge for v in graph.vertices.values() if v.type == 'scooter'])

    while unvisited_scooters:
        logging.info(f"Current station: {current_station}, Unvisited scooters: {len(unvisited_scooters)}")

        update_graph_state(graph)  # Обновляем состояние графа перед построением маршрута

        route = nearest_neighbor_route_limited(graph, current_station, num_batteries, unvisited_scooters)
        
        if len(route) < num_batteries:
            route.append(current_station)
        
        route = two_opt(route, graph)

        if route[-1] != current_station:
            nearest_station = min(
                [v.id for v in graph.vertices.values() if v.type == 'station' and v.id not in visited_stations],
                key=lambda station: graph.get_edge_weight(route[-1], station),
                default=current_station
            )
            route.append(nearest_station)
        
        stages.append(route)
        logging.info(f"Stage route: {route}")

        if route[-1] != current_station:
            current_station = route[-1]
            visited_stations.add(current_station)
        else:
            nearest_unvisited_station = min(
                [v.id for v in graph.vertices.values() if v.type == 'station' and v.id not in visited_stations],
                key=lambda station: graph.get_edge_weight(current_station, station),
                default=None
            )
            if nearest_unvisited_station:
                current_station = nearest_unvisited_station
                visited_stations.add(current_station)
            else:
                break

    update_battery_and_charge_scooters(graph, visited_stations, num_batteries, charged_scooters)
    
    avg_charge_after = np.mean([v.charge for v in graph.vertices.values() if v.type == 'scooter'])

    return stages, avg_charge_before, avg_charge_after  # Возвращаем средний заряд до и после зарядки


def two_opt(route, graph, max_iterations=1000):
    best_route = route[:]
    best_distance = total_route_distance(best_route, graph)
    iteration = 0
    improved = True

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue
                new_route = route[:]
                new_route[i:j] = reversed(route[i:j])
                new_distance = total_route_distance(new_route, graph)

                if new_distance < best_distance:
                    best_route = new_route[:]
                    best_distance = new_distance
                    improved = True

        route = best_route[:]

    return best_route

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

# Database setup
engine = create_engine('sqlite:///scooters.db')
Base.metadata.create_all(engine)

class Path:
    def __init__(self, indx, dist):
        self.indx = indx
        self.dist = dist

def main():
    num_stations = 10
    num_scooters = 300
    battery_count_per_station = 15 
    num_batteries = 15
    start_station = 's0'
    courier_speed = 1.5

    populate_database(num_stations, num_scooters, battery_count_per_station)
    graph = get_graph_from_db()
    
    stages, avg_charge_before, avg_charge_after = build_stages(graph, start_station, num_batteries)

    paths = [Path(stage, total_route_distance(stage, graph)) for stage in stages]

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

    app.run_server(debug=True)

if __name__ == "__main__":
    main()












