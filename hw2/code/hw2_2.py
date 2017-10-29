# coding: utf-8
################################################################################
# CS 224W (Fall 2017) - HW2
# Code for Problem 2.1 and 2.4
# Author: luis0@stanford.edu
# Last Updated: Oct 24, 2017
################################################################################

import snap
import random

# Problem 2.1 Functions


def loadSigns(filename):
  """
  :param - filename: undirected graph with associated edge sign

  return type: dictionary (key = node pair (a,b), value = sign)
  return: Return sign associated with node pairs. Both pairs, (a,b) and (b,a)
  are stored as keys. Self-edges are NOT included.
  """
  signs = {}
  with open(filename, 'r') as ipfile:
    for line in ipfile:
      if line[0] != '#':
        line_arr = line.split()
        if line_arr[0] == line_arr[1]:
          continue
        node1 = int(line_arr[0])
        node2 = int(line_arr[1])
        sign = int(line_arr[2])
        signs[(node1, node2)] = sign
        signs[(node2, node1)] = sign
  return signs


def getUniqueTriads(G):
  '''
  :param - G: graph

  return type: Iterator, triplets identifying unique triads (only one set per triad)
  return: Return all triads in G.
  '''
  seenTriads = {}
  for node in G.Nodes():
    neighbors = [G.GetNI(node.GetNbrNId(i))
                 for i in xrange(node.GetDeg())]
    for neighbor in neighbors:
      for i in xrange(neighbor.GetDeg()):
        candidate = neighbor.GetNbrNId(i)
        if node.IsNbrNId(candidate):
          triad = tuple(sorted(
              [node.GetId(), neighbor.GetId(), candidate]))
          if triad not in seenTriads:
            seenTriads[triad] = True
            yield triad
  assert len(seenTriads) == snap.GetTriads(G)


def computeTriadCounts(G, signs):
  """
  :param - G: graph
  :param - signs: Dictionary of signs (key = node pair (a,b), value = sign)

  return type: List, each position representing count of t0, t1, t2, and t3, respectively.
  return: Return the counts for t0, t1, t2, and t3 triad types. Count each triad
  only once and do not count self edges.
  """

  # each position represents count of t0, t1, t2, t3, respectively
  triad_count = [0, 0, 0, 0]

  ############################################################################
  mapping = {-1: 0, 1: 1}
  for triad in getUniqueTriads(G):
    index = sum([mapping[signs[(triad[i], triad[i+1])]]
                 for i in range(-1, 2)])
    triad_count[index] += 1
  assert snap.GetTriads(G) == sum(triad_count)
  ############################################################################

  return triad_count


def displayStats(G, signs):
  '''
  :param - G: graph
  :param - signs: Dictionary of signs (key = node pair (a,b), value = sign)

  Computes and prints the fraction of positive edges and negative edges,
      and the probability of each type of triad.
  '''
  fracPos = 0
  fracNeg = 0
  probs = [0, 0, 0, 0]

  ############################################################################
  fracPos = len([1 for sign in signs.values() if sign == 1]) / float(len(signs))
  fracNeg = 1 - fracPos
  probs = [fracNeg**3, 3*fracPos * fracNeg**2,
           3*fracPos**2 * fracNeg, fracPos**3]
  ############################################################################

  print 'Fraction of Positive Edges: %0.4f' % (fracPos)
  print 'Fraction of Negative Edges: %0.4f' % (fracNeg)

  for i in range(4):
    print "Probability of Triad t%d: %0.4f" % (i, probs[i])

# Problem 2.4 Functions


def createCompleteNetwork(networkSize):
  """
  :param - networkSize: Desired number of nodes in network

  return type: Graph
  return: Returns complete network on networkSize
  """
  completeNetwork = None
  ############################################################################
  completeNetwork = snap.GenFull(snap.PUNGraph, networkSize)
  ############################################################################
  return completeNetwork


def assignRandomSigns(G):
  """
  :param - G: Graph

  return type: dictionary (key = node pair (a,b), value = sign)
  return: For each edge, a sign (+, -) is chosen at random (p = 1/2).
  """
  signs = {}
  ############################################################################
  for edge in G.Edges():
    (u, v) = edge.GetId()
    sign = random.choice([1, -1])
    signs[(u, v)] = sign
    signs[(v, u)] = sign
  ############################################################################
  return signs


def runDynamicProcess(G, signs, num_iterations):
  """
  :param - G: Graph
  :param - signs: Dictionary of signs (key = node pair (a,b), value = sign)
  :param - num_iterations: number of iterations to run dynamic process

  Runs the dynamic process described in problem 2.3 for num_iterations iterations.
  """
  ############################################################################
  assert all([sign == 1 or sign == -1 for sign in signs.values()])
  BALANCED_SUMS = [-1, 3]
  triads = list(getUniqueTriads(G))
  # We assume an iteration counts even if we don't change signs.
  for _ in xrange(num_iterations):
    triad = random.choice(triads)
    edges = [(triad[i], triad[i+1]) for i in xrange(-1, 2)]
    if (sum([signs[edge] for edge in edges]) not in BALANCED_SUMS):
      (u, v) = random.choice(edges)
      assert signs[(u, v)] == signs[(v, u)]
      signs[(u, v)] *= -1
      signs[(v, u)] *= -1
  ############################################################################


def isBalancedNetwork(G, signs):
  """
  :param - G: Graph
  :param - signs: Dictionary of signs (key = node pair (a,b), value = sign)

  return type: Boolean
  return: Returns whether G is balanced (True) or not (False).
  """
  isBalanced = False
  ############################################################################
  BALANCED_SUMS = [-1, 3]
  triads = getUniqueTriads(G)
  isBalanced = all([sum([signs[(triad[i], triad[i+1])]
                         for i in xrange(-1, 2)]) in BALANCED_SUMS
                    for triad in triads])
  ############################################################################
  return isBalanced


def computeNumBalancedNetworks(numSimulations):
  """
  :param - numSimulations: number of simulations to run

  return type: Integer
  return: Returns number of networks that end up balanced.
  """
  numBalancedNetworks = 0

  for iteration in range(0, numSimulations):
    # (I) Create complete network on 10 nodes
    simulationNetwork = createCompleteNetwork(10)

    # (II) For each edge, choose a sign (+,-) at random (p = 1/2)
    signs = assignRandomSigns(simulationNetwork)

    # (III) Run dynamic process
    num_iterations = 1000000
    runDynamicProcess(simulationNetwork, signs, num_iterations)

    # determine whether network is balanced
    if isBalancedNetwork(simulationNetwork, signs):
      numBalancedNetworks += 1

  return numBalancedNetworks


def loadEpinions(filename):
  '''
  Removes self-edges from the graph!
  '''
  # load Graph and Signs
  G = snap.LoadEdgeList(snap.PUNGraph, filename, 0, 1)
  for edge in G.Edges():
    (u, v) = edge.GetId()
    if u == v:
      G.DelEdge(u, v)
  return G


def main():
  # load Graph and Signs
  filename = "data/epinions-signed.txt"
  epinionsNetwork = loadEpinions(filename)
  signs = loadSigns(filename)

  # Compute Triad Counts
  triad_count = computeTriadCounts(epinionsNetwork, signs)

  # Problem 2.1a
  print "Problem 2.1a"
  for i in range(4):
    print "Count of Triad t%d: %d" % (i, triad_count[i])

  total_triads = float(sum(triad_count)) if sum(triad_count) != 0 else 1
  for i in range(4):
    print "Fraction of Triad t%d: %0.4f" % (i, triad_count[i]/total_triads)

  # Problem 2.1b
  print "Problem 2.1b"
  displayStats(epinionsNetwork, signs)

  # Problem 2.4
  print "Problem 2.4"
  networkSize = 10
  numSimulations = 100
  numBalancedNetworks = computeNumBalancedNetworks(numSimulations)
  print("Fraction of Balanced Networks: %0.4f" % (
      float(numBalancedNetworks)/float(numSimulations)))

if __name__ == '__main__':
  main()
