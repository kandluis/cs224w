# coding: utf-8
'''
2. Further Analyzing the Wikiepedia voters networ
'''

import snap
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

SOURCE_FILE = './data/wiki-Vote.txt'
FIT_DEGREE = 1

wikiGraph = snap.LoadEdgeList(snap.PNGraph, SOURCE_FILE, 0, 1)
assert 103689 == wikiGraph.GetEdges()
assert 7115 == wikiGraph.GetNodes()

# 2.1: Histogram of out-degree distribution.
# Calculate degree and construct data frame.
outDegreeToCount = snap.TIntPrV()
snap.GetOutDegCnt(wikiGraph, outDegreeToCount)
data = pd.DataFrame([[item.GetVal1(), item.GetVal2()]
                     for item in outDegreeToCount
                     if item.GetVal1() > 0 and item.GetVal2() > 0])
data.columns = ['degree', 'count']

# Plot.
fig = plt.plot(data['degree'], data['count'], 'bo--', markersize=2)[0]
fig.axes.set_xscale('log')
fig.axes.set_yscale('log')
fig.axes.set_xlim(data['degree'].min(), data['degree'].max())
fig.axes.set_ylim(data['count'].min(), data['count'].max())
fig.axes.set_title("Log-Log Degree Distribution Plot for WikiGraph")
fig.axes.set_xlabel("Node Degree")
fig.axes.set_ylabel("Node Count")

# Save image.
plt.savefig("WikiGraphOutDegreeDistribution", format='svg', dpi=600)
plt.savefig("WikiGraphOutDegreeDistribution", dpi=600)

# Alternative 2.1.
snap.PlotOutDegDistr(wikiGraph, "WikiGraph",
                     "WikiGraph - Out Degree Distribution")

# 2.2: Compute and plot the least-square regression line.
# Calculate the best fit line on the log data.
slope, intercept = np.polyfit(
    np.log10(data['degree']), np.log10(data['count']), FIT_DEGREE)
predict = lambda x: 10**(intercept)*x**slope

# Plot.
fig = plt.plot(data['degree'], data['count'], 'bo--',
               data['degree'], predict(data['degree']), 'g',
               markersize=2)[0]
fig.axes.set_xscale('log')
fig.axes.set_yscale('log')
fig.axes.set_xlim(data['degree'].min(), data['degree'].max())
fig.axes.set_ylim(data['count'].min(), data['count'].max())
fig.axes.set_title("Log-Log Degree Distribution Plot And Fit for WikiGraph")
fig.axes.set_xlabel("Node Degree")
fig.axes.set_ylabel("Node Count")

# Save image.
plt.savefig("WikiGraphOutDegreeDistributionFit", format='svg', dpi=600)
plt.savefig("WikiGraphOutDegreeDistributionFit", dpi=600)

# Print the results.
print("The best fit line is given by equation "
      "log(y) = %s * log(x) + %s" % (slope, intercept))

# Alternative 2.1.
snap.PlotOutDegDistr(wikiGraph, "WikiGraphFit",
                     "WikiGraph - Out Degree Distribution with Fit",
                     False, True)
print("The best fit line is given by equation "
      "log(y) = -1.411 * log(x) + 3.3583 with R^2=0.91")
