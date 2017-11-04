
# coding: utf-8

# In[1]:


import snap
import numpy as np
import random
from scipy.stats import chi2_contingency


# In[2]:


def removeSelfEdges(G):
    for edge in G.Edges():
        u, v = edge.GetId()
        if u == v:
            G.DelEdge(u,v)
    return G


# In[3]:


def loadNetworks():
    '''
    loads actors, erdos, pref networks
    
    returns a dictionary of {name: network}
    '''
    actors = removeSelfEdges(snap.LoadEdgeList(
        snap.PUNGraph, "data/imdb_actor_edges.tsv", 0, 1, '\t'))
    assert actors.GetNodes() == 17577
    assert actors.GetEdges() == 287074
    erdos = removeSelfEdges(snap.LoadEdgeList(
        snap.PUNGraph, "data/SIR_erdos_renyi.txt", 0, 1, '\t'))
    assert erdos.GetNodes() == 17577
    assert erdos.GetEdges() == 287074
    pref = removeSelfEdges(snap.LoadEdgeList(
        snap.PUNGraph, "data/SIR_preferential_attachment.txt", 0, 1, '\t'))
    assert pref.GetNodes() == 17577
    assert pref.GetEdges() == 281096
    return {"Actors": actors,
            "Erdos-Renyi" : erdos,
            "Preferential Attachment" : pref}


# In[4]:


def simulation(G, infected, beta=0.05, delta=0.5):
    '''
    Simulates the SIR model of infections with the given beta and delta
    parameters.
    
    returns: the percentage of nodes that become infected.
    '''
    nodes = set([node.GetId() for node in G.Nodes()])
    susceptible = nodes - infected
    recovered = set()
    while len(infected) != 0:
        noLongerSusceptible = set()
        newlyInfected = set()
        noLongerInfected = set()
        newlyRecovered = set()
        for node in G.Nodes():
            u = node.GetId()
            if u in susceptible:
                for neighbor in [node.GetNbrNId(i)
                                 for i in xrange(node.GetDeg())]:
                    if (neighbor in infected and
                        random.random() < beta):
                        noLongerSusceptible.add(u)
                        newlyInfected.add(u)
                        break
            elif (u in infected and random.random() < delta):
                noLongerInfected.add(u)
                newlyRecovered.add(u)
                
        susceptible -= noLongerSusceptible
        infected = (infected | newlyInfected) - noLongerInfected
        recovered |= newlyRecovered
    
    return len(recovered) / float(len(nodes))


# In[5]:


def runSimulations(networks, getInitialInfected, nSimulations = 100):
    '''
    Runs the simulations and returns list for the proportion of epidemics
    in each network as well as a list of the percent infected for 
    each trial.
    
    return: dictionary with {networkname: trial_results}
    '''
    return {name: [simulation(G, getInitialInfected(G))
                   for _ in xrange(nSimulations)]
            for name, G in networks.iteritems()}


# In[6]:


def getEpidemics(results):
    '''
    Given a list of percent infected in each trial, calculates the
    proportion considered epidemics.
    
    returns proportion (float)
    '''
    return [v for v in results if v >= 0.5]


# In[7]:


def printResultStatistics(results, runSignificanceTest=True):
    for name, result in results.iteritems():
        print("\nResults for %s Network:\n" % name)
        epidemics = getEpidemics(result)
        e1 = len(epidemics)
        if len(result) == 0:
            continue
        print("%s out of %s (%s%%) simulations in %s Network resulted "
              "in an epidemic.\n" % (e1, len(result),
                                   100*float(e1) / len(result), name))
        print("On average across all trials, %s%% of the population "
              "in %s Network was infected.\n" % (100*np.mean(result), name))
        if len(epidemics) > 0:
            print("On average across epidemic trials, %s%% of the population "
                  "in %s Network was infected.\n" % (
                      100*np.mean(epidemics), name))
        else:
            print("No epidemics occurred in %s Network.\n")
    if runSignificanceTest:
        print("Significance Tests:\n")
        pairs = {}
        for name1, result1 in results.iteritems():
            for name2, result2 in results.iteritems():
                key = tuple(sorted([name1, name2]))
                if key not in pairs and name1 != name2:
                    pairs[key] = True
                    e1 = len(getEpidemics(result1))
                    e2 = len(getEpidemics(result2))
                    chi2, p, _, _ = chi2_contingency(
                        [[e1, len(result1) - e1],
                         [e2, len(result2) - e2]])
                    print("%s vs %s: chi2 = %s, p = %s.\n" % (
                        name1, name2, chi2, p))


# In[8]:


SEED = 42
Rnd = snap.TRnd(SEED)
Rnd.Randomize()
random.seed(SEED)


# In[ ]:


def Q3_1():
    '''
    Sets the global props31 and res31 variables.
    '''
    networks = loadNetworks()
    global results31
    results31 = runSimulations(networks,
                               lambda G: set([G.GetRndNId(Rnd)]))
    printResultStatistics(results31)    


# In[ ]:


Q3_1()


# In[ ]:


def Q3_2():
    '''
    Sets the global props32 and res32 variables.
    '''
    networks = loadNetworks()
    global results32
    results32 = runSimulations(networks,
                               lambda G: set([snap.GetMxDegNId(G)]))
    printResultStatistics(results32, runSignificanceTest=False)
    print("\nRelative Increases:")
    for name in results32:
        prevAvgInfected = np.mean(results31[name])
        avgInfected = np.mean(results32[name])
        relIncreases = (avgInfected - prevAvgInfected)/prevAvgInfected
        print("The average proportion infected has increased by %s%% "
              "from %s%% to %s%% for the %s Network.\n" %(
                  100*relIncreases, 100*prevAvgInfected, 100*avgInfected,
                  name))


# In[ ]:


Q3_2()


# In[ ]:


def getMaxNDegNodes(G, topN=10):
    nodeDegrees = [(node.GetId(), node.GetDeg()) for node in G.Nodes()]
    topNodeDegrees = sorted(nodeDegrees, key=lambda x: -1*x[1])[:topN]
    res = set([ID for ID, _ in topNodeDegrees])
    assert len(res) == topN
    return res


# In[ ]:


def getRandomNodes(G, Rnd, topN=10):
    res = set([])
    while len(res) < topN:
        res.add(G.GetRndNId(Rnd))
    assert len(res) == topN
    return res


# In[ ]:


def Q3_4():
    networks = loadNetworks()
    global results34_1, results34_2
    results34_1 = runSimulations(networks,
                                 lambda G: getRandomNodes(G, Rnd))
    print("Results for 10 Random Nodes:")
    printResultStatistics(results34_1, runSignificanceTest=False)
    
    print("Results for 10 Highest Degree Nodes:")
    results34_2 = runSimulations(networks,
                                 lambda G: getMaxNDegNodes(G))
    printResultStatistics(results34_2, runSignificanceTest=False)


# In[ ]:


Q3_4()

