import logging
from models import engine
from utils import populate_database, get_graph_from_db, build_stages, Path, total_route_distance
from graph import Graph
from app import create_dash_app

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Input
    num_stations = 10
    num_scooters = 150 
    battery_count_per_station = 15 
    num_batteries = 15 
    start_station = 's0'
    courier_speed = 1.5 
    # End of input

    populate_database(num_stations, num_scooters, battery_count_per_station)
    graph = get_graph_from_db()
    
    stages, avg_charge_before, avg_charge_after = build_stages(graph, start_station, num_batteries)

    paths = [Path(stage, total_route_distance(stage, graph)) for stage in stages]

    app = create_dash_app(graph, paths, avg_charge_before, avg_charge_after, courier_speed)
    app.run_server(debug=True)

if __name__ == "__main__":
    main()
