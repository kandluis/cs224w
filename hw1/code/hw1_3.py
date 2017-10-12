# coding: utf-8
################################################################################
# CS 224W (Fall 2017) - HW1
# Solution code for Problem 3.3
# Author: luis0@stanford.edu
# Last Updated: Oct 12, 2017
################################################################################

import snap
import sys
import math
import random
import matplotlib.pyplot as plt

# Setup
hT = 10
b = 2
k = 5


def sampleNodes():
  """
  return type: [[int, int]]
  return: An array of pairs of nodes
  """
  ret = []

  i = 0
  while (i < 1000):
    v = random.randint(0, (b ** hT)-1)
    w = random.randint(0, (b ** hT)-1)
    if (v != w):
      ret.append([v, w])
      i += 1

  return ret


def h(v, w):
  """
  :param - v: node id
  :param - w: node id

  return type: int
  return: h(v, w)
  """
  if (v == w):
    return 0
  else:
    xor = bin(v ^ w)[2:]
    xor = ("0" * (hT - len(xor))) + xor
    return hT - xor.find('1')


def search(Graph, s, t):
  """
  :param - s: node id
  :param - t: node id

  return type: Boolean, Int
  return: After performing the search, return either (True, <distance>) if a 
  path is found or (False, -1) otherwise.
  """

  ############################################################################
  # TODO: Your code here!
  distance = 0
  while True:
    dist = h(s, t)
    node = Graph.GetNI(s)
    distance += 1
    u, newDist = min([(node.GetOutNId(i), h(node.GetOutNId(i), t))
                      for i in xrange(node.GetOutDeg())],
                     key=lambda x: x[1])
    if u == t:
      return (True, distance)
    if dist <= newDist:
      break
    dist = newDist
    s = u
  ############################################################################

  return False, -1


def edgeProbability(alpha, v, w):
  """
  :param - alpha: given parameter [refer to 3.3]
  :param - v: node id
  :param - w: node id

  return type: Int
  return: p_v(w) [refer to 3.2]
  """
  return (b ** (-(alpha * h(v, w))))


def Z(alpha):
  """
  :param - alpha: given parameter [refer to 3.3]

  return type: Float
  return: Normalizing constant [refer to 3.2]
  """
  z = 0.0
  for i in range(1, hT+1):
    z += (pow(b, i) - pow(b, i-1)) * pow(b, -i * alpha)
  return z

from collections import defaultdict
import numpy as np


def createEdges(Graph, alpha):
  """
  :param - Graph: snap.TNGraph object representing a directed graph
  :param - alpha: given parameter [refer to 3.3]

  return type: snap.TNGraph
  return: A directed graph with edges constructed according to description
  [refer to 3.2]
  """

  ############################################################################
  # TODO: Your code here! (Hint: use Graph.AddEdge() to add edges)
  # Map from node to probabilties fo all other nodes. We assume the nodes
  # are numbered sequentially.
  constant = Z(alpha)
  probabilities = {}
  N = Graph.GetNodes()
  for i in xrange(N):
    probabilities[i] = np.zeros(N, dtype=float)
    for j in xrange(N):
      if i != j:
        probabilities[i][j] = edgeProbability(alpha, i, j)
    probabilities[i] /= float(constant)
    assert sum(probabilities[i]) - 1 < 1e-4

  for i in xrange(N):
    sources = np.random.choice(N, size=k, replace=False,
                               p=probabilities[i])
    assert len(sources) == k
    for source in sources:
      Graph.AddEdge(i, source)
  assert Graph.GetEdges() == Graph.GetNodes() * k
  for node in Graph.Nodes():
    assert node.GetOutDeg() == k
  ############################################################################

  return Graph


def runExperiment(alpha):
  """
  :param - alpha: given parameter [refer to 3.3]

  return type: [float, float]
  return: [average path length, success probability]
  """

  Graph = snap.TNGraph.New()
  for i in range(0, b ** 10):
    Graph.AddNode(i)

  Graph = createEdges(Graph, alpha)
  nodes = sampleNodes()

  c_success = 0.0
  c_path = 0.0

  for i in range(0, 1000):
    found, path = search(Graph, nodes[i][0], nodes[i][1])
    if found:
      c_success += 1
      c_path += path

  p_success = c_success/1000.0
  a_path = -1

  if c_success != 0:
    a_path = c_path/c_success

  return [a_path, p_success]


def main():
  results = []
  alpha = 0.1

  while (alpha <= 10):
    results.append([alpha] + runExperiment(alpha))
    alpha += 0.1
    print ' '.join(map(str, results[-1]))

  plt.figure(1)
  plt.plot([data[0] for data in results],
           [data[1] for data in results],
           marker='o', markersize=3)
  plt.xlabel('Parameter alpha')
  plt.ylabel('Average Path Length')
  plt.title('Average Path Length vs alpha')
  plt.savefig("output/average_path_length_vs_alpha", dpi=600)

  plt.figure(2)
  plt.plot([data[0] for data in results],
           [data[2] for data in results],
           marker='o', markersize=3)
  plt.xlabel('Parameter alpha')
  plt.ylabel('Success Rate')
  plt.title('Success Rate vs alpha')
  plt.savefig("output/average_success_vs_alpha", dpi=600)

  plt.show()


if __name__ == "__main__":
  main()
