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
In rest of this article I discuss various seed identification method and show their impact on a simulated diffusion on a publicly available Facebook ego graph (4039 nodes, 88234 edges). I also discuss the results of experiments I did to understand the diffusion behavior when <i><b>k</b></i> = 8, 13 and 21 nodes are selected as seed nodes and when network edges have probability <i><b>p</b></i> = 0.01, 0.05 and 0.1.
</br>
<h3>Data for experiments</h3>
To demonstrate the reach of initial seed nodes in information propagation in a network I use a publicly available tiny subset of Facebook ego graph from <a href="https://snap.stanford.edu/data/ego-Facebook.html">here</a>. This is an undirected graph with 4039 nodes and 88234 edges.
</br>
<h3>Why are graphs interesting</h3>
Graph data structure represents relationship information very effectively in a business setting. For example, relationships among current and potential B2B customers of a wholesaler,  a professional network of doctors who are client of a pharmaceutical company, or friends and acquaintances who are all fans of a sports team. In all these, and similar, settings nodes represent an entity - a person, an organization etc. - and edges the relationship between them. Graphs encode static relationships between these entities and graph algorithms help us understand interactions between these entities. This knowledge can be used to simulate information cascade in a network after an initial set of nodes are exposed to the information.
