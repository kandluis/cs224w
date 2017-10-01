# coding: utf-8
'''
3. Finding Experts on the Java Programming Language on StackOverflow [4 points]
 
  Download the StackOverflow network stackoverflow-Java.txt.gz:
  http://snap.stanford.edu/class/cs224w-data/hw0/stackoverflow-Java.txt.gz. 
  An edge (a, b) in the network means that person a endorsed an answer from
  person b on a Java-related question.
'''
import snap

# Load the network
SOURCE_FILE = './data/stackoverflow-Java.txt'
SOGraph = snap.LoadEdgeList(snap.PNGraph, SOURCE_FILE, 0, 1)
assert 146874 == SOGraph.GetNodes()
assert 333606 == SOGraph.GetEdges()


def sortTIntFltH(mapping, desc=True):
  return sorted([(nodeId, mapping[nodeId])
                 for nodeId in mapping
                 ], reverse=desc,
                key=lambda x: x[1])

# 3.1
components = snap.TCnComV()
snap.GetWccs(SOGraph, components)

print("The number of weakly connected components in the SO network"
      "is %s." % (len(components)))

# 3.2
maxWeaklyConnectedComponent = snap.GetMxWcc(SOGraph)
print("The largest weakly connected component in the SO network"
      "has %s nodes and %s edges." % (
          maxWeaklyConnectedComponent.GetNodes(),
          maxWeaklyConnectedComponent.GetEdges()))

# 3.3
TOPN = 3
SOPageRanks = snap.TIntFltH()
snap.GetPageRank(SOGraph, SOPageRanks, 0.85, 1e-4, 1000)
sortedSOPageRanks = sortTIntFltH(SOPageRanks)
print("The node IDs of the top %s most central nodes in the network "
      "by PageRank scores are %s with scores %s respectively." % (
          TOPN,
          tuple(t[0] for t in sortedSOPageRanks[:TOPN]),
          tuple(t[1] for t in sortedSOPageRanks[:TOPN])))


# 3.4
TOPN = 3
hubsScores = snap.TIntFltH()
authScores = snap.TIntFltH()
snap.GetHits(SOGraph, hubsScores, authScores, 100)
sortedHubScores = sortTIntFltH(hubsScores)
sortedAuthScores = sortTIntFltH(authScores)
print("The node IDs of the top %s hubs in the network by HITS scores "
      "are %s with scores %s respectively." % (
          TOPN,
          tuple(t[0] for t in sortedHubScores[:TOPN]),
          tuple(t[1] for t in sortedHubScores[:TOPN])))
print
print("The node IDs of the top %s authorities in the network by HITS "
      "scores are %s with score %s respectively." % (
          TOPN,
          tuple(t[0] for t in sortedAuthScores[:TOPN]),
          tuple(t[1] for t in sortedAuthScores[:TOPN])))
