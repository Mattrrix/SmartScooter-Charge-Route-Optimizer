from datetime import datetime

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
