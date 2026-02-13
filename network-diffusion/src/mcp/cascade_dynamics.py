from mcp.server.fastmcp import FastMCP
import json
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Tuple

mcp = FastMCP("cascade-dynamics")

# Load activation log data
with open("../../data/diffusion_stats/network_centrality_results_with_logs.json") as f:
    ACTIVATION_DATA = json.load(f)

# Load graph
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

# Load graph once at startup
G = load_graph()

# Precompute structural metrics (expensive, do once)
print("Computing structural metrics...")
DEGREE_CENTRALITY = nx.degree_centrality(G)
BETWEENNESS_CENTRALITY = nx.betweenness_centrality(G)
CLUSTERING_COEF = nx.clustering(G)
print("Structural metrics computed!")

def get_activation_log(k: int, p: float, method: str) -> Dict:
    """Helper to retrieve activation log for specific parameters"""
    try:
        return ACTIVATION_DATA["k"][str(k)]["p"][str(p)]["Mode"][method]["activation_log"]
    except KeyError:
        return {}

def get_seeds(k: int, p: float, method: str) -> List[int]:
    """Helper to retrieve seeds for specific parameters"""
    try:
        return ACTIVATION_DATA["k"][str(k)]["p"][str(p)]["Mode"][method]["seeds"]
    except KeyError:
        return []

def get_influence(k: int, p: float, method: str) -> float:
    """Helper to retrieve influence for specific parameters"""
    try:
        return ACTIVATION_DATA["k"][str(k)]["p"][str(p)]["Mode"][method]["influence"]
    except KeyError:
        return 0.0

@mcp.tool()
async def analyze_node_susceptibility(k: int, p: float, method: str, top_n: int = 20) -> str:
    """
    Identify easy vs hard to influence nodes with structural explanations.
    
    Args:
        k: Seed set size (8, 13, or 21)
        p: Propagation probability (0.01, 0.05, or 0.1)
        method: Centrality method (DEGREE_CENTRALITY, BETWEENNESS_CENTRALITY, etc.)
        top_n: Number of top nodes to return for each category (default: 20)
    
    Returns:
        JSON with easy/hard nodes and structural explanations
    """
    activation_log = get_activation_log(k, p, method)
    seeds = get_seeds(k, p, method)
    
    if not activation_log:
        return json.dumps({"error": f"No data found for k={k}, p={p}, method={method}"})
    
    # Separate seeds from non-seeds
    non_seed_nodes = []
    for node_id, stats in activation_log.items():
        node = int(node_id)
        if node not in seeds:
            non_seed_nodes.append({
                'node': node,
                'activation_probability': stats['activation_probability'],
                'avg_timestep': stats['avg_timestep'],
                'avg_attempts': stats['avg_attempts'],
                'avg_distance_from_seed': stats['avg_distance_from_seed'],
                'degree': G.degree(node),
                'degree_centrality': DEGREE_CENTRALITY[node],
                'betweenness_centrality': BETWEENNESS_CENTRALITY[node],
                'clustering_coefficient': CLUSTERING_COEF[node]
            })
    
    # Sort by activation probability (high = easy, low = hard)
    sorted_by_prob = sorted(non_seed_nodes, key=lambda x: x['activation_probability'], reverse=True)
    
    easy_nodes = sorted_by_prob[:top_n]
    hard_nodes = sorted_by_prob[-top_n:]
    
    # Calculate statistics
    easy_avg_degree = sum(n['degree'] for n in easy_nodes) / len(easy_nodes) if easy_nodes else 0
    hard_avg_degree = sum(n['degree'] for n in hard_nodes) / len(hard_nodes) if hard_nodes else 0
    
    easy_avg_clustering = sum(n['clustering_coefficient'] for n in easy_nodes) / len(easy_nodes) if easy_nodes else 0
    hard_avg_clustering = sum(n['clustering_coefficient'] for n in hard_nodes) / len(hard_nodes) if hard_nodes else 0
    
    result = {
        "parameters": {"k": k, "p": p, "method": method},
        "easy_to_influence": {
            "nodes": easy_nodes,
            "avg_degree": round(easy_avg_degree, 2),
            "avg_clustering": round(easy_avg_clustering, 4),
            "avg_activation_prob": round(sum(n['activation_probability'] for n in easy_nodes) / len(easy_nodes), 4) if easy_nodes else 0,
            "avg_timestep": round(sum(n['avg_timestep'] for n in easy_nodes) / len(easy_nodes), 2) if easy_nodes else 0
        },
        "hard_to_influence": {
            "nodes": hard_nodes,
            "avg_degree": round(hard_avg_degree, 2),
            "avg_clustering": round(hard_avg_clustering, 4),
            "avg_activation_prob": round(sum(n['activation_probability'] for n in hard_nodes) / len(hard_nodes), 4) if hard_nodes else 0,
            "avg_timestep": round(sum(n['avg_timestep'] for n in hard_nodes) / len(hard_nodes), 2) if hard_nodes else 0
        },
        "structural_explanation": {
            "degree_difference": round(easy_avg_degree - hard_avg_degree, 2),
            "clustering_difference": round(easy_avg_clustering - hard_avg_clustering, 4),
            "interpretation": "Positive degree_difference means easy nodes have more connections. Positive clustering_difference means easy nodes are in denser neighborhoods."
        }
    }
    
    return json.dumps(result, indent=2)

@mcp.tool()
async def analyze_seed_impact(k: int, p: float, method: str) -> str:
    """
    Measure which seeds caused the largest cascades.
    
    Args:
        k: Seed set size (8, 13, or 21)
        p: Propagation probability (0.01, 0.05, or 0.1)
        method: Centrality method
    
    Returns:
        JSON showing cascade size from each seed at different distances
    """
    activation_log = get_activation_log(k, p, method)
    seeds = get_seeds(k, p, method)
    
    if not activation_log or not seeds:
        return json.dumps({"error": f"No data found for k={k}, p={p}, method={method}"})
    
    # For each seed, count nodes at each distance
    seed_impact = {}
    for seed in seeds:
        seed_impact[seed] = {
            'total_influenced': 0,
            'by_distance': defaultdict(int),
            'seed_degree': G.degree(seed),
            'seed_betweenness': BETWEENNESS_CENTRALITY[seed],
            'seed_clustering': CLUSTERING_COEF[seed]
        }
    
    # Count activations by distance from each seed
    for node_id, stats in activation_log.items():
        node = int(node_id)
        if node not in seeds:  # Skip seeds themselves
            distance = stats['avg_distance_from_seed']
            prob = stats['activation_probability']
            
            # Find closest seed (approximate by checking all seeds)
            # This is simplified - in reality you'd need to track which seed activated which node
            # For now, we'll count all nodes by their distance
            for seed in seeds:
                try:
                    actual_distance = nx.shortest_path_length(G, seed, node)
                    if abs(actual_distance - distance) < 0.5:  # Close to avg distance
                        seed_impact[seed]['by_distance'][int(distance)] += prob
                        seed_impact[seed]['total_influenced'] += prob
                except nx.NetworkXNoPath:
                    pass
    
    # Sort seeds by total impact
    sorted_seeds = sorted(
        seed_impact.items(),
        key=lambda x: x[1]['total_influenced'],
        reverse=True
    )
    
    result = {
        "parameters": {"k": k, "p": p, "method": method},
        "seed_rankings": [
            {
                "seed": seed,
                "total_influenced": round(impact['total_influenced'], 2),
                "cascade_by_distance": dict(impact['by_distance']),
                "structural_properties": {
                    "degree": impact['seed_degree'],
                    "betweenness_centrality": round(impact['seed_betweenness'], 6),
                    "clustering_coefficient": round(impact['seed_clustering'], 4)
                }
            }
            for seed, impact in sorted_seeds
        ]
    }
    
    return json.dumps(result, indent=2)

@mcp.tool()
async def compare_method_reach(k: int, p: float) -> str:
    """
    Compare which method achieved best reach for given k and p.
    
    Args:
        k: Seed set size (8, 13, or 21)
        p: Propagation probability (0.01, 0.05, or 0.1)
    
    Returns:
        JSON ranking methods by influence with key differences
    """
    try:
        methods_data = ACTIVATION_DATA["k"][str(k)]["p"][str(p)]["Mode"]
    except KeyError:
        return json.dumps({"error": f"No data found for k={k}, p={p}"})
    
    method_comparison = []
    for method_name, data in methods_data.items():
        activation_log = data.get('activation_log', {})
        
        # Calculate additional metrics
        total_nodes_activated = sum(
            1 for stats in activation_log.values()
            if stats['activation_probability'] > 0.5  # Activated in >50% of simulations
        )
        
        avg_timestep_all = sum(
            stats['avg_timestep'] * stats['activation_probability']
            for stats in activation_log.values()
        ) / len(activation_log) if activation_log else 0
        
        method_comparison.append({
            "method": method_name,
            "influence": data['influence'],
            "seeds": data['seeds'],
            "total_nodes_reached": total_nodes_activated,
            "avg_activation_time": round(avg_timestep_all, 2),
            "computation_time": data['time']
        })
    
    # Sort by influence
    method_comparison.sort(key=lambda x: x['influence'], reverse=True)
    
    best_method = method_comparison[0]
    
    result = {
        "parameters": {"k": k, "p": p},
        "rankings": method_comparison,
        "best_method": {
            "name": best_method['method'],
            "influence": best_method['influence'],
            "advantage_over_second": round(
                best_method['influence'] - method_comparison[1]['influence'], 2
            ) if len(method_comparison) > 1 else 0
        }
    }
    
    return json.dumps(result, indent=2)

@mcp.tool()
async def analyze_growth_rate(k: int, p: float, method: str) -> str:
    """
    Analyze temporal growth pattern of cascade.
    
    Args:
        k: Seed set size (8, 13, or 21)
        p: Propagation probability (0.01, 0.05, or 0.1)
        method: Centrality method
    
    Returns:
        JSON with timestep distribution and growth pattern analysis
    """
    activation_log = get_activation_log(k, p, method)
    
    if not activation_log:
        return json.dumps({"error": f"No data found for k={k}, p={p}, method={method}"})
    
    # Group by timestep
    timestep_activations = defaultdict(float)
    for node_id, stats in activation_log.items():
        timestep = int(stats['avg_timestep'])
        timestep_activations[timestep] += stats['activation_probability']
    
    # Sort by timestep
    sorted_timesteps = sorted(timestep_activations.items())
    
    # Calculate growth rates between consecutive timesteps
    growth_rates = []
    for i in range(1, len(sorted_timesteps)):
        prev_total = sum(count for _, count in sorted_timesteps[:i])
        curr_total = sum(count for _, count in sorted_timesteps[:i+1])
        growth_rate = ((curr_total - prev_total) / prev_total * 100) if prev_total > 0 else 0
        growth_rates.append({
            "from_timestep": sorted_timesteps[i-1][0],
            "to_timestep": sorted_timesteps[i][0],
            "growth_rate_percent": round(growth_rate, 2)
        })
    
    # Determine pattern type
    max_timestep_count = max(count for _, count in sorted_timesteps) if sorted_timesteps else 0
    max_timestep = next((t for t, count in sorted_timesteps if count == max_timestep_count), 0)
    
    if max_timestep <= 2:
        pattern = "EXPLOSIVE - Most activations in first 2 timesteps"
    elif max_timestep <= 5:
        pattern = "RAPID - Peak activation in early timesteps (3-5)"
    else:
        pattern = "GRADUAL - Activations spread over many timesteps"
    
    result = {
        "parameters": {"k": k, "p": p, "method": method},
        "timestep_distribution": [
            {"timestep": t, "activations": round(count, 2)}
            for t, count in sorted_timesteps
        ],
        "growth_rates": growth_rates,
        "pattern_type": pattern,
        "peak_timestep": max_timestep,
        "total_timesteps": len(sorted_timesteps)
    }
    
    return json.dumps(result, indent=2)

@mcp.tool()
async def get_timestep_distribution(k: int, p: float, method: str) -> str:
    """
    Get distribution of activations by timestep.
    
    Args:
        k: Seed set size (8, 13, or 21)
        p: Propagation probability (0.01, 0.05, or 0.1)
        method: Centrality method
    
    Returns:
        JSON with detailed timestep-by-timestep breakdown
    """
    activation_log = get_activation_log(k, p, method)
    
    if not activation_log:
        return json.dumps({"error": f"No data found for k={k}, p={p}, method={method}"})
    
    # Group by timestep with more detail
    timestep_data = defaultdict(lambda: {
        'total_activations': 0,
        'nodes': [],
        'avg_attempts': 0,
        'avg_distance': 0
    })
    
    for node_id, stats in activation_log.items():
        timestep = int(stats['avg_timestep'])
        timestep_data[timestep]['total_activations'] += stats['activation_probability']
        timestep_data[timestep]['nodes'].append({
            'node': int(node_id),
            'activation_prob': stats['activation_probability'],
            'attempts': stats['avg_attempts'],
            'distance': stats['avg_distance_from_seed']
        })
        timestep_data[timestep]['avg_attempts'] += stats['avg_attempts']
        timestep_data[timestep]['avg_distance'] += stats['avg_distance_from_seed']
    
    # Calculate averages
    for timestep, data in timestep_data.items():
        num_nodes = len(data['nodes'])
        data['avg_attempts'] = round(data['avg_attempts'] / num_nodes, 2) if num_nodes > 0 else 0
        data['avg_distance'] = round(data['avg_distance'] / num_nodes, 2) if num_nodes > 0 else 0
        data['total_activations'] = round(data['total_activations'], 2)
    
    sorted_timesteps = sorted(timestep_data.items())
    
    result = {
        "parameters": {"k": k, "p": p, "method": method},
        "distribution": [
            {
                "timestep": t,
                "total_activations": data['total_activations'],
                "num_nodes": len(data['nodes']),
                "avg_attempts_at_timestep": data['avg_attempts'],
                "avg_distance_at_timestep": data['avg_distance']
            }
            for t, data in sorted_timesteps
        ]
    }
    
    return json.dumps(result, indent=2)

@mcp.tool()
async def compare_all_methods(k: int, p: float) -> str:
    """
    Comprehensive comparison of all methods for given k, p.
    
    Args:
        k: Seed set size (8, 13, or 21)
        p: Propagation probability (0.01, 0.05, or 0.1)
    
    Returns:
        JSON with detailed comparison including influence, growth patterns, and structural properties
    """
    try:
        methods_data = ACTIVATION_DATA["k"][str(k)]["p"][str(p)]["Mode"]
    except KeyError:
        return json.dumps({"error": f"No data found for k={k}, p={p}"})
    
    comparisons = []
    
    for method_name, data in methods_data.items():
        activation_log = data.get('activation_log', {})
        seeds = data['seeds']
        
        # Calculate metrics
        timestep_dist = defaultdict(float)
        total_attempts = 0
        total_distance = 0
        num_activated = 0
        
        for node_id, stats in activation_log.items():
            if int(node_id) not in seeds:  # Exclude seeds
                timestep_dist[int(stats['avg_timestep'])] += stats['activation_probability']
                total_attempts += stats['avg_attempts'] * stats['activation_probability']
                total_distance += stats['avg_distance_from_seed'] * stats['activation_probability']
                if stats['activation_probability'] > 0.5:
                    num_activated += 1
        
        # Peak timestep
        peak_timestep = max(timestep_dist.items(), key=lambda x: x[1])[0] if timestep_dist else 0
        
        # Seed structural properties
        avg_seed_degree = sum(G.degree(s) for s in seeds) / len(seeds) if seeds else 0
        avg_seed_betweenness = sum(BETWEENNESS_CENTRALITY[s] for s in seeds) / len(seeds) if seeds else 0
        
        comparisons.append({
            "method": method_name,
            "influence": data['influence'],
            "computation_time": data['time'],
            "seeds": seeds,
            "cascade_metrics": {
                "nodes_activated_over_50pct": num_activated,
                "peak_timestep": peak_timestep,
                "avg_attempts_needed": round(total_attempts / num_activated, 2) if num_activated > 0 else 0,
                "avg_cascade_depth": round(total_distance / num_activated, 2) if num_activated > 0 else 0
            },
            "seed_properties": {
                "avg_degree": round(avg_seed_degree, 2),
                "avg_betweenness": round(avg_seed_betweenness, 6)
            }
        })
    
    # Sort by influence
    comparisons.sort(key=lambda x: x['influence'], reverse=True)
    
    result = {
        "parameters": {"k": k, "p": p},
        "method_comparisons": comparisons,
        "summary": {
            "best_influence": comparisons[0]['method'] if comparisons else None,
            "fastest_computation": min(comparisons, key=lambda x: x['computation_time'])['method'] if comparisons else None,
            "fastest_cascade": min(comparisons, key=lambda x: x['cascade_metrics']['peak_timestep'])['method'] if comparisons else None
        }
    }
    
    return json.dumps(result, indent=2)

def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()