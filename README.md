# independent-cascade-diffusion-with-claude-mcp-connector
I discuss and demonstrate the impact of seed nodes selection on information diffusion in a network. I then show how insights obtained by data mining on a network can be integrated with a LLM - like Claude - by building connectors using the Model Context Protocol (MCP) architecture so that users can ask network related questions in natural language.

<b>Introduction</b>

In this article I discuss and demonstrate the impact of seed nodes selection on information diffusion in a network (or graph). Seed nodes of the graph are the initial nodes that first get exposed to the information (product launch, new social scheme etc.) and information diffusion is the distance (in number of edges) that the information travels in the network. I use classical Network Science methods - like betweenness, degree, Katz -  and Greedy Hill Climbing algorithm for identifying seed nodes.

I then show how insights obtained by data mining on a network can be integrated with a LLM - like Claude - by building connectors using the Model Context Protocol (MCP) architecture so that users can ask network related questions in natural language. These MCP connectors expands the LLM's capability by providing relevant context to answer users' questions like:

What nodes in the network are easy to influence and convert and which ones are hard?
What seeds are responsible for the largest contagion?
What method of seed identification has maximum reach when 21 nodes are used as seed nodes?
If it takes $50 to activate odd number nodes and $100 to activate even number nodes, which method is the most economical to reach at least 2000 people (where people are nodes in the graph).

Here are the links to conversation I had with Claude Desktop after integrating the results of graph data mining with the LLM:
</br>
https://claude.ai/share/c87c8469-8bda-4168-9ac2-8590f1cda668
</br>
https://claude.ai/share/885078c5-8e72-48b8-831c-3794acb6539f
