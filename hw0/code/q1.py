# coding: utf-8
'''
1. Analyzing the Wikipedia voters network [9 points]
	Download the Wikipedia voting network wiki-Vote.txt.gz: 
	http://snap.stanford.edu/data/wiki-Vote.html.

	Using one of the network analysis tools above, load the Wikipedia voting
	network. 

	Note that Wikipedia is a directed network. Formally, we consider the Wikipedia
	network as a directed graph $G = (V, E)$, with node set $V$ and edge set
	$E ⊂ V × V$ where (edges are ordered pairs of nodes). An edge $(a, b) ∈ E$
	means that user $a$ voted on user $b$.

	To make our questions clearer, we will use the following small graph as a
	running example:
	$$
	Gsmall = (Vsmall, Esmall)
	$$
	where 
	$$
	Vsmall = \{1, 2, 3\}
	$$ and
	$$
	Esmall = \{(1, 2),(2, 1),(1, 3),(1, 1)\}
	$$
'''

import snap

SOURCE_FILE = './data/wiki-Vote.txt'
DEGREE_BOUNDARY = 10

wikiGraph = snap.LoadEdgeList(snap.PNGraph, SOURCE_FILE, 0, 1)
assert 103689 == wikiGraph.GetEdges()
assert 7115 == wikiGraph.GetNodes()

# 1.1
print("The number of nodes in the network is %s." % (
    wikiGraph.GetNodes()))

# 1.2
print("The number of nodes with a self-edge is %s." % (
    snap.CntSelfEdges(wikiGraph)))

# 1.3
print("The number of directed edges %s." % (
    snap.CntUniqDirEdges(wikiGraph)))

# 1.4
print("The number of undirected edges is %s." % (
    snap.CntUniqUndirEdges(wikiGraph)))

# 1.5
print("The number of reciprocated edges is %s." % (
    snap.CntUniqDirEdges(wikiGraph) - snap.CntUniqUndirEdges(wikiGraph)))

# 1.6
print("The number of nodes of zero out-degree is %s." % (
    snap.CntOutDegNodes(wikiGraph, 0)))

# 1.7
print("The number of nodes of zero in-degree is %s." % (
    snap.CntInDegNodes(wikiGraph, 0)))

# 1.8
outDegreeToCount = snap.TIntPrV()
snap.GetOutDegCnt(wikiGraph, outDegreeToCount)
numNodesLargeOutDegree = sum([item.GetVal2()
                              for item in outDegreeToCount
                              if item.GetVal1() > DEGREE_BOUNDARY])
print("The number of nodes with more than %s outgoing edges is %s." % (
    DEGREE_BOUNDARY, numNodesLargeOutDegree))

# 1.9
inDegreeCount = snap.TIntPrV()
snap.GetOutDegCnt(wikiGraph, inDegreeCount)
numNodesSmallInDegree = sum([item.GetVal2()
                             for item in inDegreeCount
                             if item.GetVal1() < DEGREE_BOUNDARY])
print("The number of nodes with less than %s incoming edges is %s." % (
    DEGREE_BOUNDARY, numNodesSmallInDegree))
