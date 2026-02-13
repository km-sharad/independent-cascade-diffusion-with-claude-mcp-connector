from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

import random
import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

class InfluenceMaximization:
    def __init__(self, G, p=0.1, num_simulations=1000):
        self.G = G
        self.p = p
        self.num_simulations = num_simulations
    
    def simulate_ic_once(self, seeds):
        """
        ONE random cascade from FIXED seeds
        Randomness is in which edges succeed, NOT in seeds
        """
        active = set(seeds)
        newly_active = set(seeds)
        
        while newly_active:
            next_active = set()
            for u in newly_active:
                for v in self.G.neighbors(u):
                    if v not in active:
                        if random.random() < self.p:
                            next_active.add(v)
                            active.add(v)
            newly_active = next_active
        
        return len(active)
    
    def estimate_influence(self, seeds):
        """
        Estimate expected influence of FIXED seed set
        Same seeds, multiple random cascades
        """
        if not seeds:
            return 0
        
        spreads = [
            self.simulate_ic_once(seeds)
            for _ in range(self.num_simulations)
        ]
        
        return np.mean(spreads)
    
    def greedy_ic(self, k):
        seeds = set()
        current_influence = 0  # Start at 0 for empty set
        
        for iteration in range(k):
            print(f"\n=== IC Greedy: Iteration {iteration + 1}/{k} ===")
            
            best_node = None
            best_marginal = 0
            best_new_influence = 0  # Track the full influence of best choice
            
            print(f"Current seeds: {seeds}")
            print(f"Current influence: {current_influence:.2f}")

            i = 0
            for node in self.G.nodes():
                i += 1
                if node not in seeds:
                    test_seeds = seeds | {node}
                    new_influence = self.estimate_influence(test_seeds)
                    marginal = new_influence - current_influence
                    
                    if marginal > best_marginal:
                        best_marginal = marginal
                        best_node = node
                        best_new_influence = new_influence  # Save this!

                # if(i%400 == 0):
                #     print(f"iter: {iteration}, i: {i}")
            
            if best_node is None:
                print(f"Warning: No valid candidate found at iteration {iteration + 1}")
                break  # Stop early if we can't find more nodes
            
            seeds.add(best_node)
            current_influence = best_new_influence  # Use saved value, not re-estimate
            
            print(f"\n→ Selected node {best_node}")
            print(f"  Marginal gain: {best_marginal:.2f}")
        
        return list(seeds)
    
    def degree_centrality(self, k):
        """
        Select k seeds based on Degree Centrality
        (nodes with highest number of connections)
        """
        print(f"\n{'='*50}")
        print("DEGREE CENTRALITY METHOD")
        print(f"{'='*50}")
        
        # Compute degree centrality for all nodes
        degree_cent = nx.degree_centrality(self.G)
        
        # Sort nodes by degree centrality (descending)
        sorted_nodes = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 nodes by Degree Centrality:")
        for i, (node, cent) in enumerate(sorted_nodes[:10], 1):
            degree = self.G.degree(node)
            print(f"  {i}. Node {node}: centrality={cent:.4f}, degree={degree}")
        
        # Select top k nodes
        seeds = [node for node, _ in sorted_nodes[:k]]
        
        print(f"\n→ Selected seeds: {seeds}")
        
        return seeds
    
    def betweenness_centrality(self, k):
        """
        Select k seeds based on Betweenness Centrality
        (nodes that lie on many shortest paths between other nodes)
        """
        print(f"\n{'='*50}")
        print("BETWEENNESS CENTRALITY METHOD")
        print(f"{'='*50}")
        
        print("Computing betweenness centrality (this may take a while)...")
        # Compute betweenness centrality for all nodes
        betw_cent = nx.betweenness_centrality(self.G)
        
        # Sort nodes by betweenness centrality (descending)
        sorted_nodes = sorted(betw_cent.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 nodes by Betweenness Centrality:")
        for i, (node, cent) in enumerate(sorted_nodes[:10], 1):
            degree = self.G.degree(node)
            print(f"  {i}. Node {node}: centrality={cent:.4f}, degree={degree}")
        
        # Select top k nodes
        seeds = [node for node, _ in sorted_nodes[:k]]
        
        print(f"\n→ Selected seeds: {seeds}")
        
        return seeds

    def katz_centrality(self, k, alpha=0.1, beta=1.0):
        """
        Select k seeds based on Katz Centrality
        (nodes with high influence considering all paths, with exponentially 
        decaying weights for longer paths)
        
        Args:
            k: Number of top nodes to return
            alpha: Attenuation factor (default 0.1). Controls the decay of influence
                   along paths. Must be smaller than 1/lambda_max where lambda_max 
                   is the largest eigenvalue of the adjacency matrix.
            beta: Weight attributed to the immediate neighborhood (default 1.0)
        
        Returns:
            List of top k nodes by Katz centrality
        """
            
        # Get adjacency matrix as a sparse matrix
        adj_matrix = nx.adjacency_matrix(G).astype(float)
        
        # Calculate only the largest eigenvalue using sparse methods
        # k=1 means we only compute the largest eigenvalue
        largest_eigenvalue = eigsh(adj_matrix, k=1, which='LA', return_eigenvectors=False)[0]
        print(f"largest_eigenvalue: {largest_eigenvalue}")
        alpha = 0.9 / largest_eigenvalue    

        katz_cent = nx.katz_centrality(self.G, alpha=alpha, beta=beta)
        sorted_nodes = sorted(katz_cent.items(), key=lambda x: x[1], reverse=True)
        seeds = [node for node, _ in sorted_nodes[:k]]
        print(f"Katz centrality seeds: {seeds}")

        return seeds

    def geodesic_lpath_centrality(self, k):
        """
        Select top k seeds based on Geodesic L-path Centrality
        (counts the number of nodes reachable via geodesic paths of length <= l)
        
        Args:
            k: Number of top nodes to return
            l: Maximum path length to consider for centrality calculation
        
        Returns:
            List of top k nodes by geodesic l-path centrality
        """
        l = int(nx.average_shortest_path_length(self.G))
        print(f"avg path length = {l}")
        
        lpath_cent = {}
        
        # For each node, count how many nodes are reachable within l steps
        for node in self.G.nodes():
            # Get shortest path lengths from this node to all others
            lengths = nx.single_source_shortest_path_length(self.G, node, cutoff=l)
            
            # Count nodes reachable within l steps (excluding the node itself)
            lpath_cent[node] = len(lengths) - 1
        
        # Sort nodes by l-path centrality (descending)
        sorted_nodes = sorted(lpath_cent.items(), key=lambda x: x[1], reverse=True)
        
        # Select top k nodes
        seeds = [node for node, _ in sorted_nodes[:k]]
        print(f"seeds: {seeds}")
        
        return seeds

    def farness_centrality(self, k):
        """
        Select k seeds based on Farness (inverse of closeness)
        (total geodesic distance from node to all others)
        Lower values indicate MORE centrality
        """
        farness = {}
        
        for node in self.G.nodes():
            # Sum of all shortest path lengths from this node
            lengths = nx.single_source_shortest_path_length(self.G, node)
            farness[node] = sum(lengths.values())
        
        # Sort nodes by farness (ascending - lowest first for most central)
        sorted_nodes = sorted(farness.items(), key=lambda x: x[1])
        
        # Select top k nodes (those with lowest farness)
        seeds = [node for node, _ in sorted_nodes[:k]]
        
        return seeds

def compare_methods(G, k=3, p=0.1, num_simulations=1000):
    """
    Compare all three seed selection methods
    """
    print(f"\n{'#'*60}")
    print(f"INFLUENCE MAXIMIZATION COMPARISON")
    print(f"{'#'*60}")
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Seeds to select: k={k}")
    print(f"Propagation probability: p={p}")
    print(f"Simulations per evaluation: {num_simulations}")
    print(f"{'#'*60}")
    
    im = InfluenceMaximization(G, p=p, num_simulations=num_simulations)
    
    results = {}
    
    # Method 1: Degree Centrality (FAST)
    print("\n\n" + "="*60)
    print("METHOD 1: DEGREE CENTRALITY")
    print("="*60)
    start_time = time.time()
    degree_seeds = im.degree_centrality(k)
    degree_time = time.time() - start_time
    degree_influence = im.estimate_influence(degree_seeds)
    results['Degree'] = {
        'seeds': degree_seeds,
        'influence': degree_influence,
        'time': degree_time
    }
    
    # Method 2: Betweenness Centrality (MODERATE)
    print("\n\n" + "="*60)
    print("METHOD 2: BETWEENNESS CENTRALITY")
    print("="*60)
    start_time = time.time()
    betweenness_seeds = im.betweenness_centrality(k)
    betweenness_time = time.time() - start_time
    betweenness_influence = im.estimate_influence(betweenness_seeds)
    results['Betweenness'] = {
        'seeds': betweenness_seeds,
        'influence': betweenness_influence,
        'time': betweenness_time
    }

    # Method 3: Katz Centrality
    print("\n\n" + "="*60)
    print("METHOD 3: KATZ CENTRALITY")
    print("="*60)
    start_time = time.time()
    katz_seeds = im.katz_centrality(k)
    katz_time = time.time() - start_time
    katz_influence = im.estimate_influence(katz_seeds)
    results['Katz'] = {
        'seeds': katz_seeds,
        'influence': katz_influence,
        'time': katz_time
    }
    print(f"\n→ Selected seeds: {results['Katz']['seeds']}")

    # Method 4: k-geodesic Centrality
    print("\n\n" + "="*60)
    print("METHOD 4: K-GEODESIC CENTRALITY")
    print("="*60)
    start_time = time.time()
    geod_seeds = im.geodesic_lpath_centrality(k)
    geod_time = time.time() - start_time
    geod_influence = im.estimate_influence(geod_seeds)
    results['K-geodesic'] = {
        'seeds': geod_seeds,
        'influence': geod_influence,
        'time': geod_time
    }
    print(f"\n→ Selected seeds: {results['K-geodesic']['seeds']}")    

    # Method 5: Farness Centrality
    print("\n\n" + "="*60)
    print("METHOD 5: FARNESS CENTRALITY")
    print("="*60)
    start_time = time.time()
    far_seeds = im.farness_centrality(k)
    far_time = time.time() - start_time
    far_influence = im.estimate_influence(far_seeds)
    results['Farness'] = {
        'seeds': far_seeds,
        'influence': far_influence,
        'time': far_time
    }
    print(f"\n→ Selected seeds: {results['Farness']['seeds']}")        
    
    # Method 6: Greedy IC (SLOW but theoretically best)
    print("\n\n" + "="*60)
    print("METHOD 6: GREEDY INDEPENDENT CASCADE")
    print("="*60)
    start_time = time.time()
    ic_seeds = im.greedy_ic(k)
    ic_time = time.time() - start_time
    ic_influence = im.estimate_influence(ic_seeds)
    results['Greedy IC'] = {
        'seeds': ic_seeds,
        'influence': ic_influence,
        'time': ic_time
    }
    print(f"\n→ Selected seeds: {results['Greedy IC']['seeds']}")
    
    # Print comparison summary
    print("\n\n" + "#"*60)
    print("FINAL COMPARISON")
    print("#"*60)
    print(f"\n{'Method':<20} {'Seeds':<25} {'Influence':<12} {'Time (s)':<10}")
    print("-"*70)
    
    for method, data in results.items():
        seeds_str = str(data['seeds'])
        if len(seeds_str) > 24:
            seeds_str = seeds_str[:21] + "..."
        print(f"{method:<20} {seeds_str:<25} {data['influence']:<12.2f} {data['time']:<10.2f}")
    
    # Find best performer
    best_method = max(results.items(), key=lambda x: x[1]['influence'])
    fastest_method = min(results.items(), key=lambda x: x[1]['time'])
    
    print("\n" + "="*70)
    print(f"🏆 Best Influence: {best_method[0]} ({best_method[1]['influence']:.2f} nodes)")
    print(f"⚡ Fastest: {fastest_method[0]} ({fastest_method[1]['time']:.2f} seconds)")
    print("="*70)
    
    return results

def load_graph():
    """Load the Facebook graph from edge list"""
    file_path = "../../data/facebook_ego/facebook_combined.txt"
    edges = []
    with open(file_path, 'r') as f:
        for line in f:
            nodes = [int(x) for x in line.strip().split()]
            edges.append(tuple(nodes))
    
    G = nx.Graph()
    G.add_edges_from(edges)
    return G

if __name__ == "__main__":
    G = load_graph()
      
    # Run comparison
    results = compare_methods(
        G, 
        k=8,                    # Number of seeds to find
        p=0.01,                  # Propagation probability
        num_simulations=50     # Simulations per evaluation (reduce for speed)
    )