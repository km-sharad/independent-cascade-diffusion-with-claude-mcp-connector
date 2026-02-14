# independent-cascade-diffusion-with-claude-mcp-connector
I discuss and demonstrate the impact of seed nodes selection on information diffusion in a network. Using a publicly available Facebook ego graph, I then show how insights obtained by data mining on a network can be integrated with a LLM - like Claude - by building connectors using the <a href="https://code.claude.com/docs/en/mcp">Model Context Protocol (MCP)</a> architecture so that users can ask network related questions in natural language.

<h3>Introduction</h3>
In this article I discuss and demonstrate the impact of seed nodes selection on information diffusion in a network (or graph). Seed nodes of the graph are the initial nodes that first get exposed to the information (product launch, new social scheme etc.) and information diffusion is the distance (in number of edges) that the information travels in the network. I use classical Network Science centrality measures - like betweenness, degree, Katz -  and <a href="https://www.cs.cornell.edu/home/kleinber/icalp05-inf.pdf">Greedy Hill Climbing algorithm</a>, due to Kempe <i>et al.</i>, for identifying seed nodes. I use the <a href="https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf">Independent Cascade</a> diffusion algorithm, also due to Kempe <i>et al.</i>, for simulating diffusion in the network.
</br>
</br>
I then show how insights obtained by data mining on a network can be integrated with a LLM - like Claude - by building connectors using the Model Context Protocol (MCP) architecture so that users can ask network related questions in natural language. These MCP connectors extend the LLM's capability by providing relevant context to answer users' questions like:
</br>
<ul>
<li>What nodes in the network are easier to influence and convert and which ones are harder?</li>
<li>What seeds are responsible for the largest contagion?</li>
<li>What method of seed identification has maximum reach when 21 nodes are used as seed nodes?</li>
<li>If it takes $50 to activate odd number nodes and $100 to activate even number nodes, which method is the most economical to reach at least 2000 people?</li>
</ul>

Here are the links to conversation I had with Claude Desktop after integrating the results of graph data mining with the LLM:
</br>
https://claude.ai/share/c87c8469-8bda-4168-9ac2-8590f1cda668
</br>
https://claude.ai/share/885078c5-8e72-48b8-831c-3794acb6539f
</br>
</br>
In rest of this article I discuss various seed identification method and show their impact on a simulated diffusion on a publicly available Facebook ego graph (4039 nodes, 88234 edges). I also discuss the results of experiments I did to understand the diffusion behavior when <i><b>k</b></i> = 8, 13 and 21 nodes are selected as seed nodes and when network edges have probability <i><b>p</b></i> = 0.01, 0.05 and 0.1 of getting activated.
</br>
<h3>Data for experiments</h3>
To demonstrate the reach of initial seed nodes in information propagation in a network I use a publicly available tiny subset of Facebook ego graph from <a href="https://snap.stanford.edu/data/ego-Facebook.html">here</a>. This is an undirected graph with 4039 nodes and 88234 edges.
</br>
<h3>Why are graphs interesting</h3>
Graph data structure represents relationship information very effectively in a business setting. For example, relationships among current and potential B2B customers of a wholesaler,  a professional network of doctors who are client of a pharmaceutical company, or friends and acquaintances who are all fans of a sports team. In all these, and similar, settings nodes represent an entity - a person, an organization etc. - and edges the relationships between them. Graphs encode static relationships between these entities and graph algorithms help us understand interactions between these entities. This knowledge can be used to simulate information cascade in a network after an initial set of nodes are exposed to the information.
</br>
<h3>Seed selection and diffusion simulation methods</h3>
Optimal selection of the seed nodes is important because the right selection of these node exploits the structure of the underlying network for  diffusion of information organically. I discuss two  techniques for identifying seed nodes:

1. Classical network science centrality measures: betweenness, degree, Katz etc.
2. Greedy Hill Climbing algorithm using Independent Cascade (IC) diffusion method as proposed by Kempe *et al.* [^kempe]

I briefly explain these measures below. Jackson [^Jackson] and Barabási [^Barabasi] have covered these topics in their books at length.

<i>Classical Network Science based node selection methods:</i>
</br>
DEGREE CENTRALITY: ratio of number of nodes incident of a node over (n-1), where n is the total number of nodes in the network. I take <i><b>k</i></b> nodes with highest degree centrality measure as seeds.
</br>
BETWEENNESS CENTRALITY: the number of times that any node needs a given node to reach any other node. I take <i><b>k</i></b> nodes with highest betweenness centrality measure as seeds.
</br>
KATZ CENTRALITY: sums all walks starting or ending at a node, regardless of length; an attenuation factor here makes shorter  paths more valuable than longer ones. I take <i><b>k</i></b> nodes with highest Katz centrality measure as seeds.
</br>
K-GEODESIC CENTRALITY: the number of geodesic paths (the shortest path between two nodes) up to length <i><b>k</i></b> emanating from a given node. I take <i><b>k</i></b> nodes with highest K-Geodesic centrality measure as seeds. (it's unfortunate that I am using k for max number of edges in a geodesic and also for number of seeds, just to be clear - they mean different things) 
</br>
FARNESS CENTRALITY: the total geodesic distance from a given node to all other nodes. Closeness centrality is an inverse measure of centrality since larger values indicate less centrality. I take <i><b>k</i></b> nodes with lowest farness centrality measure as seeds.
</br>
</br>
<i>Greedy Hill Climbing Algorithm:</i>
</br>
The Greedy Hill Climbing algorithm is based upon the principle of submodularity also know as diminishing returns condition. In this context it means that a node when added to the larger set of seeds will result in lower gain, in terms of reach, as compared to when it's added to a smaller set of seeds. Kempe <i>et al.</i> discuss this in their paper <i>Influential Nodes in a Diffusion Model for Social Networks</i>. [^kempe2] Here's is the algorithm from the paper (<i><b>k</b></i> is the number of seeds to be identified):
</br>
```
start with A = ∅ 
for i = 1 to k do 
	let vi be a node (approximately) maximizing the marginal gain σ(A ∪ {vi}) − σ(A).
	Set A ← A ∪ {vi}. 
end for 
```  

I use the Independent Cascade diffusion simulation method [^kempe], to calculate the marginal gain in the first line of the for loop above.
</br>
</br>
<i>Information diffusion simulation algorithm</i>
</br>
Other than using the IC method for identifying nodes in the Hill Climbing algorithm, I also use it to simulate diffusion in the network from seed nodes identified by the methods above. The IC method is a probabilistic algorithm where once a node is activated, it gets one chance to activate all its neighbors with a predefined probability <i><b>p</b></i>. (In their paper Kempe et al. use a different activation probability for each node, for simplicity I use a fixed value for <i><b>p</b></i>). This algorithm progresses in time steps and nodes activated in one time step attempt to activate their neighbors in the next time step. Details about the IC and Greedy Hill Climbing algorithm are in the paper[^kempe2].
</br>
<h3>Code</h3>
<i>Seed identification and diffusion simulation code</i>
</br>
The code for identifying the seed nodes and simulating the diffusion in the network is in `network-diffusion/src/diffusion/fb_ego_diffusion_with_node_timesteps.py`. Given the initial number of seed nodes <i><b>k</b></i> and the probability <i><b>p</b></i> of a node activating its neighbor, I generate the following statistics for all seed identification methods:
<ul>
<li>Influence or reach in terms of total number of nodes activated</li>
<li>Time steps taken for a node to get activated</li>
</ul>

The code runs 50 simulations of the diffusion and collect the following statistics:
<ul>
<li>Activation probability of a node = the number of times the node got activated/50 </li>
<li>Average timestep of activation = average number of timesteps taken to activate the node in iterations where the node got activated</li>
<li>Average attempts = average number of attempts taken to activate the node</li>
<li>Average distance from the seed node (in iterations when the node got activated)</li>
</ul>
The statistics generated by the diffusion algorithm is saved in a JSON file and is available in the codebase here: `network-diffusion/data/diffusion_stats/network_centrality_results_with_logs.json`. Next, I explain how the statistics available in this file is used by the MCP connector.

<h3>Integration with Claude Desktop using Model Context Protocol (MCP)</h3>
MCP (Model Context Protocol) is an open-source protocol created by Anthropic that enables AI assistants like Claude to connect with external data sources and tools. I created MCP servers that use the statistics available in JSON file with diffusion statistics to obtain context knowledge for answering user questions about the graph. I connect these servers to Claude Desktop app using its connector pattern to integrate with external source. 
</br>
<i>MCP Code</i>
</br>
Code for MCP servers is available `network-diffusion/src/mcp/cascade_dynamics.py` and `network-diffusion/src/mcp/network_centrality.py`.
</br>
<h3>Experiments and results:</h3>
To compare the influence spread by seeds identified by different techniques I simulated the diffusion on <b><i>k</i></b> = 8, 13 and 21 initial nodes and activation probability values of <b><i>p</i></b> = 0.01, 0.05 and 0.1.  Whereas the method of identifying seed nodes is formula based, the Hill Climbing node identification algorithm aborts if the reach does not improve in a timestep.
</br>
In my experiments I found that in almost all the cases while nodes identified by betweenness centrality performed best and slightly better than the nodes identified by the Greedy Hill Climbing algorithm, the influence spread (no. of nodes activated) to number of seed nodes ratio was usually better for the Greedy Hill Climbing algorithm. Results of the experiments are shown in the table below.
</br>
</br>

| k  | p    | Mode                    | Numbers of seed nodes activated | Influence (number of nodes reached by diffusion) | Time to find k nodes (secs) |
|----|------|-------------------------|----------------------------------|--------------------------------------------------|-----------------------------|
| 8  | 0.01 | DEGREE CENTRALITY       | 8  | 295.54  | 0.00    |
| 8  | 0.01 | BETWEENNESS CENTRALITY  | 8  | 218.90  | 99.97   |
| 8  | 0.01 | KATZ CENTRALITY         | 8  | 189.04  | 11.88   |
| 8  | 0.01 | K-GEODESIC CENTRALITY   | 8  | 95.34   | 38.76   |
| 8  | 0.01 | FAIRNESS CENTRALITY     | 8  | 111.92  | 31.21   |
| 8  | 0.01 | GREEDY IC               | 8  | 282.18  | 6838.37 |
| 8  | 0.05 | DEGREE CENTRALITY       | 8  | 2156.88 | 0.00    |
| 8  | 0.05 | BETWEENNESS CENTRALITY  | 8  | 2199.02 | 99.35   |
| 8  | 0.05 | KATZ CENTRALITY         | 8  | 1784.38 | 11.80   |
| 8  | 0.05 | K-GEODESIC CENTRALITY   | 8  | 1958.86 | 39.02   |
| 8  | 0.05 | FAIRNESS CENTRALITY     | 8  | 2002.50 | 31.61   |
| 8  | 0.05 | GREEDY IC               | 8  | 2206.40 | 25451.75|
| 8  | 0.1  | DEGREE CENTRALITY       | 8  | 2955.62 | 0.00    |
| 8  | 0.1  | BETWEENNESS CENTRALITY  | 8  | 3044.02 | 98.64   |
| 8  | 0.1  | KATZ CENTRALITY         | 8  | 2886.86 | 11.88   |
| 8  | 0.1  | K-GEODESIC CENTRALITY   | 8  | 2935.02 | 38.47   |
| 8  | 0.1  | FAIRNESS CENTRALITY     | 8  | 2900.28 | 30.86   |
| 8  | 0.1  | GREEDY IC               | 8  | 3017.70 | 29968.03|
| 13 | 0.01 | DEGREE CENTRALITY       | 13 | 318.68  | 0.00    |
| 13 | 0.01 | BETWEENNESS CENTRALITY  | 13 | 245.22  | 99.30   |
| 13 | 0.01 | KATZ CENTRALITY         | 13 | 189.78  | 11.80   |
| 13 | 0.01 | K-GEODESIC CENTRALITY   | 13 | 105.10  | 39.17   |
| 13 | 0.01 | FAIRNESS CENTRALITY     | 13 | 134.90  | 31.49   |
| 13 | 0.01 | GREEDY IC               | 9  | 232.46  | 7919.42 |
| 13 | 0.05 | DEGREE CENTRALITY       | 13 | 2164.50 | 0.00    |
| 13 | 0.05 | BETWEENNESS CENTRALITY  | 13 | 2203.12 | 99.01   |
| 13 | 0.05 | KATZ CENTRALITY         | 13 | 1853.48 | 11.68   |
| 13 | 0.05 | K-GEODESIC CENTRALITY   | 13 | 1873.42 | 38.71   |
| 13 | 0.05 | FAIRNESS CENTRALITY     | 13 | 1956.96 | 31.26   |
| 13 | 0.05 | GREEDY IC               | 7  | 2205.48 | 25664.30|
| 13 | 0.1  | DEGREE CENTRALITY       | 13 | 2962.48 | 0.00    |
| 13 | 0.1  | BETWEENNESS CENTRALITY  | 13 | 3033.34 | 98.92   |
| 13 | 0.1  | KATZ CENTRALITY         | 13 | 2941.80 | 11.72   |
| 13 | 0.1  | K-GEODESIC CENTRALITY   | 13 | 2926.00 | 38.81   |
| 13 | 0.1  | FAIRNESS CENTRALITY     | 13 | 2919.00 | 31.40   |
| 13 | 0.1  | GREEDY IC               | 7  | 3039.42 | 24448.87|
| 21 | 0.01 | DEGREE CENTRALITY       | 21 | 337.94  | 0.00    |
| 21 | 0.01 | BETWEENNESS CENTRALITY  | 21 | 289.26  | 98.83   |
| 21 | 0.01 | KATZ CENTRALITY         | 21 | 185.30  | 11.80   |
| 21 | 0.01 | K-GEODESIC CENTRALITY   | 21 | 145.22  | 38.13   |
| 21 | 0.01 | FAIRNESS CENTRALITY     | 21 | 130.86  | 30.84   |
| 21 | 0.01 | GREEDY IC               | 9  | 269.30  | 7710.43 |
| 21 | 0.05 | DEGREE CENTRALITY       | 21 | 2153.28 | 0.00    |
| 21 | 0.05 | BETWEENNESS CENTRALITY  | 21 | 2232.28 | 98.64   |
| 21 | 0.05 | KATZ CENTRALITY         | 21 | 1756.68 | 11.70   |
| 21 | 0.05 | K-GEODESIC CENTRALITY   | 21 | 2012.40 | 38.31   |
| 21 | 0.05 | FAIRNESS CENTRALITY     | 21 | 2043.70 | 30.83   |
| 21 | 0.05 | GREEDY IC               | 14 | 2231.92 | 51047.54|
| 21 | 0.1  | DEGREE CENTRALITY       | 21 | 2941.88 | 0.00    |
| 21 | 0.1  | BETWEENNESS CENTRALITY  | 21 | 3051.24 | 99.93   |
| 21 | 0.1  | KATZ CENTRALITY         | 21 | 2876.14 | 11.85   |
| 21 | 0.1  | K-GEODESIC CENTRALITY   | 21 | 2912.36 | 39.17   |
| 21 | 0.1  | FAIRNESS CENTRALITY     | 21 | 2932.46 | 31.62   |
| 21 | 0.1  | GREEDY IC               | 6  | 3054.28 | 24119.13|

[^kempe]: <a href="https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf">Kempe, D., Kleinberg, J., & Tardos, É. (2003). Maximizing the spread of influence through a social network.</a>
[^Jackson]: <a href="https://press.princeton.edu/books/paperback/9780691148205/social-and-economic-networks?srsltid=AfmBOooobgqQMQIZoMb-Pk4aMhmodIGw2s_FLBCuqqM3AWnlpp8Kd5zh">Jackson, M (2008). Social and Economic Networks.</a>
[^Barabasi]: <a href="https://networksciencebook.com/">Barabasi, A (2016). Network Science.</a>
[^kempe2]: <a href="https://www.cs.cornell.edu/home/kleinber/icalp05-inf.pdf">Kempe, D., Kleinberg, J., & Tardos, É. (2005). Influential Nodes in a Diffusion Model for Social Networks.</a>
