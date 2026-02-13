from mcp.server.fastmcp import FastMCP
import json

mcp = FastMCP("network-centrality")

# Load your data once at startup
with open("../../data/diffusion_stats/network_data.json") as f:
    # /Users/thematrix/sharadkumar/Code/Git/independent-cascade-diffusion-with-claude-mcp-connector/network-diffusion/data/diffusion_stats/network_data.json
    NETWORK_DATA = json.load(f)

@mcp.tool()
async def get_centrality_results(
    k: int,
    p: float, 
    method: str
) -> str:
    """Get centrality results for specific parameters.
    
    Args:
        k: Seed set size (8, 13, or 21)
        p: Propagation probability (0.01, 0.05, or 0.1)
        method: Centrality method (DEGREE_CENTRALITY, BETWEENNESS_CENTRALITY, 
                KATZ_CENTRALITY, K_GEODESIC_CENTRALITY, FARNESS_CENTRALITY, GREEDY_IC)
    """
    try:
        result = NETWORK_DATA["k"][str(k)]["p"][str(p)]["Mode"][method]
        
        # Add metadata about early termination
        requested_k = k
        actual_k = len(result["seeds"])
        
        return json.dumps({
            **result,
            "requested_k": requested_k,
            "actual_k": actual_k,
            "early_termination": actual_k < requested_k
        }, indent=2)
    except KeyError:
        return f"Error: No data found for k={k}, p={p}, method={method}"

@mcp.tool()
async def compare_methods(k: int, p: float) -> str:
    """Compare all centrality methods for given parameters.
    
    Args:
        k: Seed set size (8, 13, or 21)
        p: Propagation probability (0.01, 0.05, or 0.1)
    """
    try:
        methods = NETWORK_DATA["k"][str(k)]["p"][str(p)]["Mode"]
        
        comparison = []
        for method_name, data in methods.items():
            comparison.append({
                "method": method_name,
                "influence": data["influence"],
                "time": data["time"],
                "num_seeds": len(data["seeds"]),
                "requested_seeds": k
            })
        
        # Sort by influence
        comparison.sort(key=lambda x: x["influence"], reverse=True)
        
        return json.dumps(comparison, indent=2)
    except KeyError:
        return f"Error: No data found for k={k}, p={p}"

@mcp.tool()
async def find_best_method(
    k: int,
    p: float,
    optimize_for: str = "influence"
) -> str:
    """Find the best performing method.
    
    Args:
        k: Seed set size (8, 13, or 21)
        p: Propagation probability (0.01, 0.05, or 0.1)
        optimize_for: Optimization criterion ("influence" or "time")
    """
    try:
        methods = NETWORK_DATA["k"][str(k)]["p"][str(p)]["Mode"]
        
        if optimize_for == "influence":
            best = max(methods.items(), key=lambda x: x[1]["influence"])
        elif optimize_for == "time":
            best = min(methods.items(), key=lambda x: x[1]["time"])
        else:
            return "Error: optimize_for must be 'influence' or 'time'"
        
        method_name, data = best
        return json.dumps({
            "best_method": method_name,
            "optimized_for": optimize_for,
            **data,
            "actual_seeds": len(data["seeds"]),
            "requested_seeds": k
        }, indent=2)
    except KeyError:
        return f"Error: No data found for k={k}, p={p}"

@mcp.tool()
async def get_seed_nodes(k: int, p: float, method: str) -> str:
    """Get the seed node IDs for a specific centrality method.
    
    Args:
        k: Seed set size (8, 13, or 21)
        p: Propagation probability (0.01, 0.05, or 0.1)
        method: Centrality method (DEGREE_CENTRALITY, BETWEENNESS_CENTRALITY, 
                KATZ_CENTRALITY, K_GEODESIC_CENTRALITY, FARNESS_CENTRALITY, GREEDY_IC)
    """
    try:
        result = NETWORK_DATA["k"][str(k)]["p"][str(p)]["Mode"][method]
        
        return json.dumps({
            "method": method,
            "k": k,
            "p": p,
            "seeds": result["seeds"],
            "num_seeds": len(result["seeds"]),
            "requested_seeds": k
        }, indent=2)
    except KeyError:
        return f"Error: No data found for k={k}, p={p}, method={method}"

@mcp.tool()
async def analyze_influence_performance(k: int, p: float, method: str) -> str:
    """Analyze influence performance of a centrality method across different parameters.
    
    Args:
        k: Seed set size (8, 13, or 21)
        p: Propagation probability (0.01, 0.05, or 0.1)
        method: Centrality method to analyze
    """
    try:
        result = NETWORK_DATA["k"][str(k)]["p"][str(p)]["Mode"][method]
        
        # Compare with other methods at same k, p
        methods = NETWORK_DATA["k"][str(k)]["p"][str(p)]["Mode"]
        all_influences = [(name, data["influence"]) for name, data in methods.items()]
        all_influences.sort(key=lambda x: x[1], reverse=True)
        
        # Find rank
        rank = next(i for i, (name, _) in enumerate(all_influences, 1) if name == method)
        
        # Calculate relative performance
        max_influence = all_influences[0][1]
        relative_performance = (result["influence"] / max_influence) * 100 if max_influence > 0 else 0
        
        return json.dumps({
            "method": method,
            "k": k,
            "p": p,
            "influence": result["influence"],
            "rank": rank,
            "total_methods": len(all_influences),
            "relative_to_best": f"{relative_performance:.2f}%",
            "best_method": all_influences[0][0],
            "best_influence": all_influences[0][1],
            "all_rankings": all_influences
        }, indent=2)
    except KeyError:
        return f"Error: No data found for k={k}, p={p}, method={method}"

@mcp.tool()
async def analyze_time_performance(k: int, p: float, method: str) -> str:
    """Analyze computational time performance of a centrality method.
    
    Args:
        k: Seed set size (8, 13, or 21)
        p: Propagation probability (0.01, 0.05, or 0.1)
        method: Centrality method to analyze
    """
    try:
        result = NETWORK_DATA["k"][str(k)]["p"][str(p)]["Mode"][method]
        
        # Compare with other methods at same k, p
        methods = NETWORK_DATA["k"][str(k)]["p"][str(p)]["Mode"]
        all_times = [(name, data["time"]) for name, data in methods.items()]
        all_times.sort(key=lambda x: x[1])
        
        # Find rank (1 = fastest)
        rank = next(i for i, (name, _) in enumerate(all_times, 1) if name == method)
        
        # Calculate relative performance
        fastest_time = all_times[0][1]
        if fastest_time > 0:
            slowdown_factor = result["time"] / fastest_time
        else:
            slowdown_factor = 1.0 if result["time"] == 0 else float('inf')
        
        return json.dumps({
            "method": method,
            "k": k,
            "p": p,
            "time_seconds": result["time"],
            "rank": rank,
            "total_methods": len(all_times),
            "slowdown_vs_fastest": f"{slowdown_factor:.2f}x",
            "fastest_method": all_times[0][0],
            "fastest_time": all_times[0][1],
            "all_rankings": all_times
        }, indent=2)
    except KeyError:
        return f"Error: No data found for k={k}, p={p}, method={method}"

@mcp.tool()
async def get_convergence_info(k: int, p: float, method: str) -> str:
    """Get convergence information including early termination details.
    
    Args:
        k: Seed set size (8, 13, or 21)
        p: Propagation probability (0.01, 0.05, or 0.1)
        method: Centrality method to analyze
    """
    try:
        result = NETWORK_DATA["k"][str(k)]["p"][str(p)]["Mode"][method]
        
        requested_k = k
        actual_k = len(result["seeds"])
        early_termination = actual_k < requested_k
        
        termination_reason = None
        if early_termination:
            if method == "GREEDY_IC":
                termination_reason = "Greedy algorithm stopped early - no marginal gain from adding more seeds"
            else:
                termination_reason = "Method returned fewer seeds than requested"
        
        return json.dumps({
            "method": method,
            "k": k,
            "p": p,
            "requested_seeds": requested_k,
            "actual_seeds": actual_k,
            "early_termination": early_termination,
            "seeds_difference": requested_k - actual_k,
            "termination_reason": termination_reason,
            "influence": result["influence"],
            "time": result["time"],
            "seed_nodes": result["seeds"]
        }, indent=2)
    except KeyError:
        return f"Error: No data found for k={k}, p={p}, method={method}"
def main():
    # Initialize and run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()    