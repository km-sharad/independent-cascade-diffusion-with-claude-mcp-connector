# independent-cascade-diffusion-with-claude-mcp-connector
I discuss and demonstrate the impact of seed nodes selection on information diffusion in a network. I then show how insights obtained by data mining on a network can be integrated with a LLM - like Claude - by building connectors using the Model Context Protocol (MCP) architecture so that users can ask network related questions in natural language.

<h3>Introduction</h3>
In this article I discuss and demonstrate the impact of seed nodes selection on information diffusion in a network (or graph). Seed nodes of the graph are the initial nodes that first get exposed to the information (product launch, new social scheme etc.) and information diffusion is the distance (in number of edges) that the information travels in the network. I use classical Network Science centrality measures - like betweenness, degree, Katz -  and <a href="https://www.cs.cornell.edu/home/kleinber/icalp05-inf.pdf">Greedy Hill Climbing algorithm</a>, due to Kempe <i>et al.</i> for identifying seed nodes. I use the <a href="https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf">Independent Cascade</a> diffusion algorithm, also due to Kempe <i>et al.</i>, for simulating diffusion in the network.
</br>
</br>
I then show how insights obtained by data mining on a network can be integrated with a LLM - like Claude - by building connectors using the Model Context Protocol (MCP) architecture so that users can ask network related questions in natural language. These MCP connectors expands the LLM's capability by providing relevant context to answer users' questions like:
</br>
<ul>
<li>What nodes in the network are easy to influence and convert and which ones are hard?</li>
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
Graph data structure represents relationship information very effectively in a business setting. For example, relationships among current and potential B2B customers of a wholesaler,  a professional network of doctors who are client of a pharmaceutical company, or friends and acquaintances who are all fans of a sports team. In all these, and similar, settings nodes represent an entity - a person, an organization etc. - and edges the relationship between them. Graphs encode static relationships between these entities and graph algorithms help us understand interactions between these entities. This knowledge can be used to simulate information cascade in a network after an initial set of nodes are exposed to the information.
</br>
<h3>Seed selection and diffusion simulation methods</h3>
Optimal selection of the seed nodes is important because the right selection of these node exploits the structure of the underlying network for  diffusion of information organically. I discuss two  techniques for identifying seed nodes:

1. Classical network science centrality measures: betweenness, degree, Katz etc.
2. Greedy Hill Climbing algorithm using Independent Cascade (IC) diffusion method as proposed by Kempe *et al.* [^kempe]

I briefly explain these measures below. Jackson [^Jackson] and Barabasi [^Barabasi] have explained these in their books.

<i>Classical Network Science based node selection methods:</i>
</br>
DEGREE CENTRALITY: ratio of number of nodes incident of a node over (n-1), where n is the total number of nodes in the network. I take k nodes with highest degree centrality measure as seeds.
</br>
BETWEENNESS CENTRALITY: the number of times that any node needs a given node to reach any other node. I take k nodes with highest betweenness centrality measure as seeds.
</br>
KATZ CENTRALITY: sums all walks starting or ending at a node, regardless of length; an attenuation factor here makes shorter  paths more valuable than longer ones. For k seeds, I take k nodes with highest Katz centrality measure as seeds.
</br>
K-GEODESIC CENTRALITY: the number of geodesic paths (the shortest path between two nodes) up to length k emanating from a given node. For k seeds, I take k nodes with highest K-Geodesic centrality measure as seeds. (it's unfortunate that I am using k for max number of edges in a geodesic and also for number of seeds, just to be clear - they mean different things) 
</br>
FARNESS CENTRALITY: the total geodesic distance from a given node to all other nodes. Closeness centrality is an inverse measure of centrality since larger values indicate less centrality. I take k nodes with lowest farness centrality measure as seeds.
</br>
<i>Greedy Hill Climbing Algorithm:</i>
</br>
The Greedy Hill Climbing algorithm is based upon the principle of submodularity also know as diminishing returns condition. In this context it means that a node when added to the larger set of seeds will result in lower gain, in terms of reach, as compared to when it's added to a smaller set of seeds. Kempe <i>et al.</i> discuss this in their paper <i>Influential Nodes in a Diffusion Model for Social
Networks</i>. [^kempe2] Here's is the algorithm from the paper (<i><b>k</b></i> is the number of seeds to be identified):
</br>
```
	start with A = ∅ 
	for I = 1 to k do 
		let vi be a node (approximately) maximizing the marginal gain σ(A ∪ {vi}) − σ(A).
		Set A ← A ∪ {vi}. 
	end for 
```  







[^kempe]: Kempe, D., Kleinberg, J., & Tardos, É. (2003). Maximizing the spread of influence through a social network.
[^Jackson]: Jackson, M (2008). Social and Economic Networks.
[^Barabasi]: Barabasi, A (2016). Network Science.
[^kempe2]: Kempe, D., Kleinberg, J., & Tardos, É. (2005). Influential Nodes in a Diffusion Model for Social Networks
