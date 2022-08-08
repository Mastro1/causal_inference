# Causal Inference

According to Wikipedia, Causal inference is *"the process of determining the independent, actual effect of a particular phenomenon that is a component of a larger system"*. 
Here the project realized for the course at [EPFL](https://www.epfl.ch).

## Exercise 1 - Learning the causal structure of a Bayesian network
For the first exercise, we coded the three functions **SGS**, **PC1**, and **PC2**, so that they are ready to be applied to our four different datasets.

Next, we integrated into our code an algorithm allowing us to partially orient a graph starting from the two main outputs of Task 1: the unoriented graphs, and the separating sets for all non-adjacent variables. The algorithm first exploits the graph’s v-structures and then applies the first two Meek’s rules in a loop.

Finally, we plotted the oriented graphs. The results of D1, D2, and D3 are visible in the Figure below, which shows nine Bayesian graphs oriented according to SGS, PC1, and PC2 algorithms. For computational reasons, we couldn’t display the graph nor the adjacency matrix for the dataset D4.

<p align="center"><img src="https://drive.google.com/uc?id=1hTX5M_EGTbxfveq7neeWcjk7Nz9ZzPO-" width="900"/></p>

Although statistically consistent, the SGS algorithm is computationally inefficient. 

The number of conditional independence tests it does grows exponentially as a function of the number of variables analyzed, each represented by a node in the graph. The SGS algorithm starts from the worst case, without exploiting any possible, statistically sufficient short-cuts.
The SGS algorithm’s complexity is O(2n), with n being the number of variables represented in the graphs, that is, an exponential search. The upper bound of CI tests to perform (worst case scenario) according to SGS is 2^n [[1]](#1).

PC logic is like the SGS algorithm. The two algorithms also rely on the same assumptions, yet the PC algorithm is much more efficient as it needs to perform fewer statistical tests. Hence, it operates much faster. PC algorithms’ computational complexity is O(n^d max), with n being the number of variables represented in the graph, and dmax being the maximal degree of the graph [[2]](#2), which typically grows with the number of nodes *n* [[3]](#3). The upper bound of CI tests to perform (worst case scenario) according to PCs is n^d max.
In practice, implementing the three algorithms with the three datasets, we had to perform the number of CI tests resumed in Table below. However, they do not always correspond to our expectations with SGS not necessarily being less efficient than PC.

|                | D1 | D2  | D3   |
|----------------|----|-----|------|
| SGS - CI tests | 46 | 538 | 7351 |
| PC1 - CI tests | 51 | 551 | 5681 |
| PC2 - CI tests | 42 | 472 | 5398 |


## Exercise 2 - Analysing the stock market data
For this question, we tried to build a Bayesian network showing the causal structure between 12 public companies listed in the table below along with their ticker symbol and an index to trace them within the graphs. Our objective was to infer the existence of pairwise directed information (DI) going from one company’s stock price at time t-1, to another company’s stock price at time t, while conditioning for the price movements of all other 10 stocks.

| Name                          | Ticker | Index |
|-------------------------------|--------|-------|
| AppleInc.                     | APPL   | (1)   |
| CiscoSystemsInc.              | CSCO   | (2)   |
| DellInc. Inc.                 | DEL    | (3)   |
| EMCCorporation                | EMC    | (4)   |
| GoogleInc.                    | GOG    | (5)   |
| Hewlett-Packard               | HP     | (6)   |
| Intel                         | INT    | (7)   |
| Microsoft                     | MSFT   | (8)   |
| Oracle                        | ORC    | (9)   |
| InternationalBusinessMachines | IBM    | (10)  |
| TexasInstruments              | TXN    | (11)  |
| Xerox                         | XRX    | (12)  |

Assuming the underlying dynamic which describes the relationships between the log-prices of these companies to be jointly Gaussian, we computed the DI 𝐼(𝑥 → 𝑦 || 𝑧) as required.
Then validated the existence of a DI, represented by the edge within the graph, iff 𝐼(𝑥 → 𝑦 || 𝑧) > 0.5.

The algorithm led us to the causal structure reported in Figure below.

<p align="center"><img src="https://drive.google.com/uc?id=1Mye6MX56y7EGhhbGcLhbMtI0-lpiq9EL" width="300"/></p>


As a final step, we implemented the same algorithm to learn a causal structure that is less demanding in terms of DI threshold and therefore allows us to infer more DIs. Even though the new DIs may be considered less robust because of the threshold reduction, they could provide valuable and reliable information if the threshold is rigorously diminished.
Our algorithm set a new threshold obtained by convergence. We start with an arbitrary maximum and minimum threshold of 0.75 and 0.25 respectively, we average these two extremes and set an average threshold accordingly (0.5 in the first case).

We then verify if using this average threshold, the DIs found are such that we can connect all of the 12 nodes to at least one other. If that is the case, the minimum threshold is replaced by the average of the previous threshold’s extremes (0.75 and 0.25, in the first case). If this is not the case, the maximum threshold is replaced by the average of the previous threshold’s extremes (0.75 and 0.25, in the first case).
We then iterate this process 100 times. After roughly 50 iterations, the algorithm already finds a stable DI-threshold value. And after all the 100 iterations, the final threshold is set to 0.38153419677443867, and it allows to link all the nodes at least into couples.
Hence, we conserve the highest threshold allowing at least pairwise edges for all the nodes.

Implementing the new threshold within the algorithm yields the causal structure reported in Figure below.

Ultimately, one should note that bi-directional arrows are NOT an indication of node directions: in our code, they are used to plot un-directed edges.

<p align="center"><img src="https://drive.google.com/uc?id=1OLsLAwAHHg-jzVAE-hlsDQ4jdEQ9offY" width="300"/></p>


## References
<a id="1">[1]</a> Glymour, C., Spirtes, P., & Scheines, R. (1991). Causal Inference. Erkenntnis (1975-), 35(1/3), 151–189. http://www.jstor.org/stable/20012366 <br>
<a id="2">[2]</a> The Maximum Degree of G denoted by Δ(G), is the degree of the vertex with the greatest number of edges incident to it <br>
<a id="3">[3]</a> Sondhi, A., & Shojaie, A. (2019). The Reduced PC-Algorithm: Improved Causal Structure Learning in Large Random Networks. J. Mach. Learn. Res., 20(164), 1-31


