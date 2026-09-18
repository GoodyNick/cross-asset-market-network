# Quantitative Cross-Asset Network Topology & Systemic Risk Analysis

An automated data pipeline and statistical physics model evaluating dynamic market fragility, asset decoupling, and systemic contagion across traditional macro and digital asset classes.

## Executive Summary

Standard Pearson correlation matrices often obscure true economic relationships due to finite-sample noise and transitive clustering. This project applies Random Matrix Theory (RMT) and Graph Theory to filter empirical noise and extract the structural backbone of the global market. By tracking the Minimum Spanning Tree (MST) over rolling windows, it generates a time-varying Systemic Stress Index that anticipates liquidity contractions and measures Bitcoin's regime shifts between equity beta and decoupled store of value.

## Methodologies & Statistical Physics

* **Mantegna Metric Space:** Maps raw correlations into an ultrametric distance ($d_{ij} = \sqrt{2(1 - \rho_{ij})}$) to enforce triangle inequality, extracting the network's Minimum Spanning Tree via Kruskal's algorithm.


* **Random Matrix Theory (RMT):** Decomposes the empirical correlation matrix and evaluates the eigenvalue spectrum against theoretical Marčenko-Pastur bounds to separate global market modes and structural signals from random noise.
* **Rolling Systemic Stress Index ($L(t)$):** Tracks the normalized Total Tree Length over 60-day sliding windows, identifying periods of systemic contagion where cross-asset distances collapse ($d \to 0$).

## Key Quantitative Findings

* **Spurious Hub Demotion:** In raw empirical networks, Crude Oil appears as a dominant degree-3 hub connecting equities, currencies, and gold. RMT spectral cleaning proves this is a statistical artifact, establishing the S&P 500 as the true systemic anchor.


* **The Bitcoin Beta Regime:** Topologically, Bitcoin attaches exclusively to the S&P 500 with the shortest distance in the network ($d = 1.12$, $\rho \approx +0.37$), acting strictly as a high-beta liquidity asset rather than an inflation hedge during the 2021–2025 period.


* **Topological Invariants:** The defensive duration branch (Crude Oil $\to$ Gold $\to$ 10Y Treasury) persists through noise filtering, confirming that the bond-yield and precious-metal corridor acts as an invariant macroeconomic law.



## Technical Architecture

* **Data Engineering:** `pandas`, `yfinance`, `parquet` storage protocols.
* **Network Science:** `NetworkX` for graph construction and MST extraction.


* **Numerical Methods:** `numpy` and `scipy` for spectral decomposition and eigenvalue clumping.
* **Visualization:** `matplotlib` and `seaborn` for dynamic two-panel stress tracking and side-by-side RMT network comparisons.



---

This README structure explicitly translates your computational physics scripts into the exact commercial signals outlined in your My first 200$ outcome strategy. By highlighting the RMT matrix algebra and the actionable trading insights, you immediately demonstrate to quantitative trading desks and Web3 founders that your technical expertise generates tangible financial intelligence. What specific subreddits or Discord communities will you target first with this repository link?