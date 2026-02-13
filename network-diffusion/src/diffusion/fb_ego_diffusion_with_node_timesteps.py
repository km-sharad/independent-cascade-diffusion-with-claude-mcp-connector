import random
import networkx as nx
import numpy as np
import time
import json
from collections import defaultdict

class InfluenceMaximization:
    def __init__(self, G, p=0.1, num_simulations=1000):
        self.G = G
        self.p = p
        self.num_simulations = num_simulations
    
    def simulate_ic_once_with_log(self, seeds):
        """
        ONE random cascade from FIXED seeds with detailed activation logging
        Returns: (num_activated, activation_log)
        """
        activation_log = {}
        active = set(seeds)
        newly_active = set(seeds)
        
        # Track distance from nearest seed for each node
        distance_from_seed = {}
        
        # Initialize seeds at timestep 0 with distance 0
        # Use sorted order for deterministic processing
        for seed in sorted(seeds):
            activation_log[seed] = {
                'timestep': 0,
                'activated_by': None,
                'attempts': 0,
                'distance_from_seed': 0
            }
            distance_from_seed[seed] = 0
        
        timestep = 1
        attempt_counts = defaultdict(int)
        
        while newly_active:
            next_active = set()
            # Process nodes in sorted order for reproducibility
            for u in sorted(newly_active):
                # Process neighbors in sorted order
                for v in sorted(self.G.neighbors(u)):
                    if v not in active:
                        attempt_counts[v] += 1
                        if random.random() < self.p:
                            next_active.add(v)
                            active.add(v)
                            v_distance = distance_from_seed[u] + 1
                            distance_from_seed[v] = v_distance
                            activation_log[v] = {
                                'timestep': timestep,
                                'activated_by': u,
                                'attempts': attempt_counts[v],
                                'distance_from_seed': v_distance
                            }
            newly_active = next_active
            timestep += 1
        
        return len(active), activation_log
    
    def estimate_influence_with_logs(self, seeds):
        """
        Estimate expected influence and collect aggregated activation logs
        Returns: (mean_influence, aggregated_activation_data)
        """
        if not seeds:
            return 0, {}
        
        spreads = []
        all_logs = []
        
        for _ in range(self.num_simulations):
            spread, log = self.simulate_ic_once_with_log(seeds)
            spreads.append(spread)
            all_logs.append(log)
        
        # Aggregate activation statistics across all simulations
        node_stats = defaultdict(lambda: {
            'activation_count': 0,
            'total_timesteps': 0,
            'total_attempts': 0,
            'total_distance': 0
        })
        
        for log in all_logs:
            for node, info in log.items():
                node_stats[node]['activation_count'] += 1
                node_stats[node]['total_timesteps'] += info['timestep']
                node_stats[node]['total_attempts'] += info['attempts']
                node_stats[node]['total_distance'] += info['distance_from_seed']
        
        # Calculate averages
        aggregated_log = {}
        for node, stats in node_stats.items():
            count = stats['activation_count']
            aggregated_log[str(node)] = {
                'activation_probability': round(count / self.num_simulations, 4),
                'avg_timestep': round(stats['total_timesteps'] / count, 2) if count > 0 else 0,
                'avg_attempts': round(stats['total_attempts'] / count, 2) if count > 0 else 0,
                'avg_distance_from_seed': round(stats['total_distance'] / count, 2) if count > 0 else 0,
                'times_activated': count
            }
        
        return np.mean(spreads), aggregated_log
    
    def greedy_ic(self, k):
        seeds = []  # Use list instead of set
        current_influence = 0
        activation_log = {}
        
        for iteration in range(k):
            print(f"\n=== IC Greedy: Iteration {iteration + 1}/{k} ===")
            
            best_node = None
            best_marginal = 0
            best_new_influence = 0
            best_log = {}
            
            print(f"Current seeds: {seeds}")
            print(f"Current influence: {current_influence:.2f}")

            i = 0
            for node in self.G.nodes():
                i += 1
                if node not in seeds:
                    test_seeds = seeds + [node]  # List concatenation
                    new_influence, temp_log = self.estimate_influence_with_logs(test_seeds)
                    marginal = new_influence - current_influence
                    
                    if marginal > best_marginal:
                        best_marginal = marginal
                        best_node = node
                        best_new_influence = new_influence
                        best_log = temp_log

                if(i%400 == 0):
                    print(f"iter: {iteration}, i: {i}")            
            
            if best_node is None:
                print(f"Warning: No valid candidate found at iteration {iteration + 1}")
                break
            
            seeds.append(best_node)  # Append to list
            current_influence = best_new_influence
            activation_log = best_log
            
            print(f"\n→ Selected node {best_node}")
            print(f"  Marginal gain: {best_marginal:.2f}")
        
        return seeds, activation_log
    
    def degree_centrality(self, k):
        """
        Select k seeds based on Degree Centrality
        """
        print(f"\n{'='*50}")
        print("DEGREE CENTRALITY METHOD")
        print(f"{'='*50}")
        
        degree_cent = nx.degree_centrality(self.G)
        sorted_nodes = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 nodes by Degree Centrality:")
        for i, (node, cent) in enumerate(sorted_nodes[:10], 1):
            degree = self.G.degree(node)
            print(f"  {i}. Node {node}: centrality={cent:.4f}, degree={degree}")
        
        seeds = [node for node, _ in sorted_nodes[:k]]
        print(f"\n→ Selected seeds: {seeds}")
        
        return seeds
    
    def betweenness_centrality(self, k):
        """
        Select k seeds based on Betweenness Centrality
        """
        print(f"\n{'='*50}")
        print("BETWEENNESS CENTRALITY METHOD")
        print(f"{'='*50}")
        
        print("Computing betweenness centrality (this may take a while)...")
        betw_cent = nx.betweenness_centrality(self.G)
        sorted_nodes = sorted(betw_cent.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 nodes by Betweenness Centrality:")
        for i, (node, cent) in enumerate(sorted_nodes[:10], 1):
            degree = self.G.degree(node)
            print(f"  {i}. Node {node}: centrality={cent:.4f}, degree={degree}")
        
        seeds = [node for node, _ in sorted_nodes[:k]]
        print(f"\n→ Selected seeds: {seeds}")
        
        return seeds

    def katz_centrality(self, k, alpha=0.1, beta=1.0):
        """
        Select k seeds based on Katz Centrality
        """
        from scipy.sparse.linalg import eigsh
        
        adj_matrix = nx.adjacency_matrix(self.G).astype(float)
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
        """
        l = int(nx.average_shortest_path_length(self.G))
        print(f"avg path length = {l}")
        
        lpath_cent = {}
        
        for node in self.G.nodes():
            lengths = nx.single_source_shortest_path_length(self.G, node, cutoff=l)
            lpath_cent[node] = len(lengths) - 1
        
        sorted_nodes = sorted(lpath_cent.items(), key=lambda x: x[1], reverse=True)
        seeds = [node for node, _ in sorted_nodes[:k]]
        print(f"seeds: {seeds}")
        
        return seeds

    def farness_centrality(self, k):
        """
        Select k seeds based on Farness (inverse of closeness)
        """
        farness = {}
        
        for node in self.G.nodes():
            lengths = nx.single_source_shortest_path_length(self.G, node)
            farness[node] = sum(lengths.values())
        
        sorted_nodes = sorted(farness.items(), key=lambda x: x[1])
        seeds = [node for node, _ in sorted_nodes[:k]]
        
        return seeds

def compare_methods(G, k_values, p_values, num_simulations=1000):
    """
    Compare all methods across different k and p values
    Returns nested dictionary structure matching the desired JSON format
    """
    results = {"k": {}}
    
    for k in k_values:
        results["k"][str(k)] = {"p": {}}
        
        for p in p_values:
            print(f"\n{'#'*60}")
            print(f"INFLUENCE MAXIMIZATION COMPARISON")
            print(f"{'#'*60}")
            print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            print(f"Seeds to select: k={k}")
            print(f"Propagation probability: p={p}")
            print(f"Simulations per evaluation: {num_simulations}")
            print(f"{'#'*60}")
            
            im = InfluenceMaximization(G, p=p, num_simulations=num_simulations)
            
            results["k"][str(k)]["p"][str(p)] = {"Mode": {}}
            mode_results = results["k"][str(k)]["p"][str(p)]["Mode"]
            
            # Method 1: Degree Centrality
            print("\n\n" + "="*60)
            print("METHOD 1: DEGREE CENTRALITY")
            print("="*60)
            start_time = time.time()
            degree_seeds = im.degree_centrality(k)
            degree_time = time.time() - start_time
            degree_influence, degree_log = im.estimate_influence_with_logs(degree_seeds)
            mode_results['DEGREE_CENTRALITY'] = {
                'seeds': degree_seeds,
                'influence': round(degree_influence, 2),
                'time': round(degree_time, 2),
                'activation_log': degree_log
            }
            
            # Method 2: Betweenness Centrality
            print("\n\n" + "="*60)
            print("METHOD 2: BETWEENNESS CENTRALITY")
            print("="*60)
            start_time = time.time()
            betweenness_seeds = im.betweenness_centrality(k)
            betweenness_time = time.time() - start_time
            betweenness_influence, betweenness_log = im.estimate_influence_with_logs(betweenness_seeds)
            mode_results['BETWEENNESS_CENTRALITY'] = {
                'seeds': betweenness_seeds,
                'influence': round(betweenness_influence, 2),
                'time': round(betweenness_time, 2),
                'activation_log': betweenness_log
            }

            # Method 3: Katz Centrality
            print("\n\n" + "="*60)
            print("METHOD 3: KATZ CENTRALITY")
            print("="*60)
            start_time = time.time()
            katz_seeds = im.katz_centrality(k)
            katz_time = time.time() - start_time
            katz_influence, katz_log = im.estimate_influence_with_logs(katz_seeds)
            mode_results['KATZ_CENTRALITY'] = {
                'seeds': katz_seeds,
                'influence': round(katz_influence, 2),
                'time': round(katz_time, 2),
                'activation_log': katz_log
            }
            print(f"\n→ Selected seeds: {mode_results['KATZ_CENTRALITY']['seeds']}")

            # Method 4: K-geodesic Centrality
            print("\n\n" + "="*60)
            print("METHOD 4: K-GEODESIC CENTRALITY")
            print("="*60)
            start_time = time.time()
            geod_seeds = im.geodesic_lpath_centrality(k)
            geod_time = time.time() - start_time
            geod_influence, geod_log = im.estimate_influence_with_logs(geod_seeds)
            mode_results['K_GEODESIC_CENTRALITY'] = {
                'seeds': geod_seeds,
                'influence': round(geod_influence, 2),
                'time': round(geod_time, 2),
                'activation_log': geod_log
            }
            print(f"\n→ Selected seeds: {mode_results['K_GEODESIC_CENTRALITY']['seeds']}")    

            # Method 5: Farness Centrality
            print("\n\n" + "="*60)
            print("METHOD 5: FARNESS CENTRALITY")
            print("="*60)
            start_time = time.time()
            far_seeds = im.farness_centrality(k)
            far_time = time.time() - start_time
            far_influence, far_log = im.estimate_influence_with_logs(far_seeds)
            mode_results['FARNESS_CENTRALITY'] = {
                'seeds': far_seeds,
                'influence': round(far_influence, 2),
                'time': round(far_time, 2),
                'activation_log': far_log
            }
            print(f"\n→ Selected seeds: {mode_results['FARNESS_CENTRALITY']['seeds']}")        
            
            # Method 6: Greedy IC
            print("\n\n" + "="*60)
            print("METHOD 6: GREEDY INDEPENDENT CASCADE")
            print("="*60)
            start_time = time.time()
            ic_seeds, ic_log = im.greedy_ic(k)
            ic_time = time.time() - start_time
            ic_influence, _ = im.estimate_influence_with_logs(ic_seeds)  # Re-estimate for consistency
            mode_results['GREEDY_IC'] = {
                'seeds': ic_seeds,
                'influence': round(ic_influence, 2),
                'time': round(ic_time, 2),
                'activation_log': ic_log
            }
            print(f"\n→ Selected seeds: {mode_results['GREEDY_IC']['seeds']}")
            
            # Print summary for this k, p combination
            print("\n\n" + "#"*60)
            print(f"SUMMARY FOR k={k}, p={p}")
            print("#"*60)
            print(f"\n{'Method':<25} {'Influence':<12} {'Time (s)':<10}")
            print("-"*50)
            
            for method, data in mode_results.items():
                print(f"{method:<25} {data['influence']:<12.2f} {data['time']:<10.2f}")
    
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
    # Load your graph
    G = load_graph()
    
    # Define parameter ranges
    k_values = [8, 13, 21]
    p_values = [0.01, 0.05, 0.1]
    
    # Run comparison
    results = compare_methods(
        G, 
        k_values=k_values,
        p_values=p_values,
        num_simulations=50
    )
    
    # Save to JSON file
    output_filename = 'network_centrality_results_with_logs.json'
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\n{'='*60}")
    print(f"Results saved to {output_filename}")
    print(f"{'='*60}")