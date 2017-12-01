# coding: utf-8
import os

import snap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def readLabels():
  filename = "data/polblogs-labels.txt"
  with open(filename) as data:
    return np.array(map(lambda x: int(x) if int(x) == 1 else -1, list(data)))


def readAdjacencyGraph(labels):
  filename = "data/polblogs.txt"
  N = len(labels)
  matrix = np.zeros((N, N))
  with open(filename) as data:
    for line in data:
      ixs = map(int, line.strip().split(" "))
      assert len(ixs) == 2
      i, j = ixs[0], ixs[1]
      matrix[i][j] = 1
      matrix[j][i] = 1
  return matrix


def minimizationAlgorithm(A):
  D = np.diag(np.sum(A, axis=0))
  invD = np.diag(np.sum(A, axis=0)**-1)
  L = D - A
  normL = np.dot(np.sqrt(invD), np.dot(L, np.sqrt(invD)))
  w, V = np.linalg.eigh(normL)
  assert abs(w[0] - 0) < 1e-9
  assert w[1] > 0
  v = V[:, 1]
  v.shape = (len(v), 1)
  x = np.sqrt(np.sum(A)) * np.dot(np.sqrt(invD), v)
  assert abs(np.dot(x.T, np.dot(D, x)) - np.sum(A)) < 1e-7
  e = np.ones(len(D))
  e.shape = (len(e), 1)
  assert abs(np.dot(x.T, np.dot(D, e))) < 1e-8
  assignments = np.sign(x)
  assert sum(assignments == 0) == 0
  return assignments


def maximAlgorithm(A):
  d = np.sum(A, axis=0)
  d.shape = (len(d), 1)
  B = A - 1.0 / np.sum(A) * np.dot(d, d.T)

  w, V = np.linalg.eigh(B)
  v = V[:, -1]
  v.shape = (len(v), 1)
  y = np.sqrt(len(A)) * v
  assert abs(np.dot(y.T, y) - len(A)) < 1e-7
  assignments = np.sign(y)
  assert sum(assignments == 0) == 0
  return assignments


def Q4_4():
  G = snap.LoadEdgeList(
      snap.PUNGraph, "data/polblogs.txt", 0, 1, ' ')
  labels = readLabels()
  # We verify the nodes match as expected.
  assert sorted([n.GetId() for n in G.Nodes()]) == range(len(labels))

  network = readAdjacencyGraph(labels)

  assignments1 = minimizationAlgorithm(network).flatten()
  assignments2 = maximAlgorithm(network).flatten()

  print "Minimization gives #nodes in S as %s and nodes in \bar{S} as %s." % (
      sum(assignments1 == 1), sum(assignments1 == -1))
  print "Maximization gives #nodes in S as %s and nodes in \bar{S} as %s." % (
      sum(assignments2 == 1), sum(assignments2 == -1))

  print "The accuracy of the first assignment is %s." % (
      sum(assignments1 == labels) / float(len(labels)))
  print "The accuracy of the first assignment is %s." % (
      sum(assignments2 == -1*labels) / float(len(labels)))


Q4_4()
