# coding: utf-8
import os

import numpy as np
import matplotlib.pyplot as plt


def readThresholdDistribution():
  filename = "data/thresholds.txt"
  with open(filename) as data:
    return map(int, list(data))


def Q1a():
  dist = readThresholdDistribution()
  X = range(len(dist))
  Y = np.cumsum(dist)

  plt.close()
  plt.title("Cumulative Histogram of Thresholds")
  plt.xlabel("Threshold")
  plt.ylabel("Total nodes with equal or smaller threshold")
  plt.plot(X, Y)
  plt.plot(X, X)
  if not os.path.exists("output"):
    os.mkdir("output")
  plt.savefig("output/1a", dpi=500)
  plt.show()

  maxRiots = 0
  for i, y in enumerate(Y):
    if y <= i:
      maxRiots = i
      break
  print("The final number of rioters is %s." % maxRiots)

Q1a()
