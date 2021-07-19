import random

import networkx as nx
import numpy as np

class RWGraphs():
    def __init__(self, nx_Gs, nx_G, node_type=None):
        self.Gs = nx_Gs
        self.G = nx_G
        self.node_type = node_type

    def walk(self, walk_length, start, gs, schema=None):
        # Simulate a random walk starting from start node.
        rand = random.Random()

        if schema:
            schema_items = schema.split('-')

        walk = [start]
        edges = []
        while len(walk) < walk_length:
            cur = walk[-1]
            pickup = random.choice([(typeid, g) for typeid, g in enumerate(gs) if cur in g.nodes()]) # at least one
            G = pickup[1]
            typeid = pickup[0]
            candidates = []
            for node in G[cur].keys():
                if schema == None or self.node_type[node] == schema_items[len(walk) % (len(schema_items) - 1)]:
                    candidates.append(node)
            if candidates:
                walk.append(rand.choice(candidates))
                edges.append(typeid)
            else:
                break
        return [walk, edges]

    def simulate_walks(self, num_walks, walk_length, schema=None):
        G = self.G
        walks = []
        nodes = list(G.nodes())
        # print('Walk iteration:')
        if schema is not None:
            schema_list = schema.replace(' ', '').split(',')
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                if schema is None:
                    walks.append(self.walk(walk_length=walk_length, start=node, gs=self.Gs))
                else:
                    for idx, schema_iter in enumerate(schema_list):
                        if schema_iter.split('-')[0] == self.node_type[node]:
                            # 该部分增加索引，标识该路径是通过关系idx得到的采样结果
                            walks.append((idx, self.walk(walk_length=walk_length, start=node, gs=self.Gs, schema=schema_iter)))
                            # walks.append(self.walk(walk_length=walk_length, start=node, schema=schema_iter))
        return walks

class RWGraph():
    def __init__(self, nx_G, node_type=None):
        self.G = nx_G
        self.node_type = node_type

    def walk(self, walk_length, start, schema=None):
        # Simulate a random walk starting from start node.
        G = self.G

        rand = random.Random()

        if schema:
            schema_items = schema.split('-')

        walk = [start]
        while len(walk) < walk_length:
            cur = walk[-1]
            candidates = []
            for node in G[cur].keys():  # U - I - U
                if schema == None or self.node_type[node] == schema_items[len(walk) % (len(schema_items) - 1)]:
                    candidates.append(node)
            if candidates:
                walk.append(rand.choice(candidates))
            else:
                break
        return [int(node) for node in walk]

    def simulate_walks(self, num_walks, walk_length, schema=None):
        G = self.G
        walks = []
        nodes = list(G.nodes())
        # print('Walk iteration:')
        if schema is not None:
            schema_list = schema.replace(' ', '').split(',')
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                if schema is None:
                    walks.append(self.walk(walk_length=walk_length, start=node))
                else:
                    for idx, schema_iter in enumerate(schema_list):
                        if schema_iter.split('-')[0] == self.node_type[node]:
                            # 该部分增加索引，标识该路径是通过关系idx得到的采样结果
                            walks.append((idx, self.walk(walk_length=walk_length, start=node, schema=schema_iter)))
                            # walks.append(self.walk(walk_length=walk_length, start=node, schema=schema_iter))
        return walks


