#include "hnsw.h"

#include <algorithm>
#include <iostream>
#include <queue>
#include <random>
#include <set>
#include <unordered_set>
#include <vector>
#include <omp.h>
using namespace std;



vector<int> HNSWGraph::searchLayer(Item& q, int ep, int ef, int lc) {
    // cout << "[*] Start searchLayer" << endl;
    set<pair<double, int>> candidates;
    set<pair<double, int>> nearestNeighbors;
    unordered_set<int> isVisited;
    vector<int> results;

    double td = q.dist(items[ep]);
    candidates.insert(make_pair(td, ep));
    nearestNeighbors.insert(make_pair(td, ep));
    isVisited.insert(ep);
    
    // cout << "[*] Start searchLayer parallel" << endl;
    
    while (!candidates.empty()) {
        auto ci = candidates.begin(); 
        candidates.erase(ci);
        int nid = ci->second;
        auto fi = nearestNeighbors.rbegin();
        if (ci->first > fi->first) {
            break;
        }

        for (int ed : layerEdgeLists[lc][nid]) {
            #pragma omp task firstprivate(q, ed, ef) shared(isVisited, candidates, nearestNeighbors, items)
            {
                bool visited = false;
                #pragma omp critical
                {
                    visited = isVisited.find(ed) != isVisited.end();
                    if (!visited) 
                        isVisited.insert(ed);
                    // cout << "visited: " << visited << endl;
                }
                // cout << "visited: " << visited << endl;
                if (!visited) {
                    double local_td = q.dist(items[ed]);
                    bool insert_candidate = false;
                    #pragma omp critical
                    {
                        auto local_fi = nearestNeighbors.end();
                        local_fi--;
                        if ((local_td < local_fi->first) || nearestNeighbors.size() < ef) {
                            insert_candidate = true;
                            nearestNeighbors.insert(make_pair(local_td, ed));
                            if (nearestNeighbors.size() > ef) {
                                nearestNeighbors.erase(local_fi);
                            }
                        }
                    }
                    if (insert_candidate) {
                        #pragma omp critical
                        {
                            candidates.insert(make_pair(local_td, ed));
                        }
                    }
                }
            }
        }
        #pragma omp taskwait
    }

    // cout << "[*] Set searchLayer Result" << endl;
    for (auto& p : nearestNeighbors) {
        results.push_back(p.second);
    }

    // cout << "[*] Done searchLayer" << endl;
    return results;
}



vector<int> HNSWGraph::KNNSearch(Item& q, int K) {
	// cout << "[*] Start KNNSearch" << endl;
	int maxLyer = layerEdgeLists.size() - 1;
	int ep = enterNode;
    vector<int> result;
    #pragma omp parallel num_threads(40) shared(ep, q, K, result)
    {
        #pragma omp single
        {
            for (int l = maxLyer; l >= 1; l--)
                ep = searchLayer(q, ep, 1, l)[0];
            result = searchLayer(q, ep, K, 0);
        }
    }

	// for (int l = maxLyer; l >= 1; l--) 
	// 	ep = searchLayer(q, ep, 1, l)[0];
	// vector<int> result = searchLayer(q, ep, K, 0);

	// cout << "[*] Done KNNSearch" << endl;
	return result;
}


void HNSWGraph::addEdge(int st, int ed, int lc) {
	if (st == ed) return;
	#pragma omp critical
	{
		layerEdgeLists[lc][st].push_back(ed);
		layerEdgeLists[lc][ed].push_back(st);
	}
}


void HNSWGraph::Insert(Item& q) {
	// cout << "[*] Start Insert" << endl;
	int nid = items.size();
	itemNum++; items.push_back(q);
	// sample layer
	int maxLyer = layerEdgeLists.size() - 1;
	int l = 0;
	uniform_real_distribution<double> distribution(0.0,1.0);

	while(l < ml && (1.0 / ml <= distribution(generator))) {
		l++;
		if (layerEdgeLists.size() <= l) 
			layerEdgeLists.push_back(unordered_map<int, vector<int>>());
	}

	if (nid == 0) {
		enterNode = nid;
		return;
	}

	// search up layer entrance
    int ep = enterNode;
    #pragma omp parallel num_threads(40) shared(ep, q)
    {
        #pragma omp single
        {
            for (int i = maxLyer; i > l; i--) 
                ep = searchLayer(q, ep, 1, i)[0];
        }
    }
    // for (int i = maxLyer; i > l; i--) 
    //     ep = searchLayer(q, ep, 1, i)[0];
	
	for (int i = min(l, maxLyer); i >= 0; i--) {
		int MM = l == 0 ? MMax0 : MMax;
		vector<int> neighbors = searchLayer(q, ep, efConstruction, i);
		vector<int> selectedNeighbors = vector<int>(neighbors.begin(), neighbors.begin()+min(int(neighbors.size()), M));

		// 병렬화 고려
		for (int n: selectedNeighbors)
			addEdge(n, nid, i);

		for (int n: selectedNeighbors) {
			if (layerEdgeLists[i][n].size() > MM) {
				vector<pair<double, int>> distPairs;
				for (int nn: layerEdgeLists[i][n])
					distPairs.emplace_back(items[n].dist(items[nn]), nn);
				sort(distPairs.begin(), distPairs.end());
				layerEdgeLists[i][n].clear();
				for (int d = 0; d < min(int(distPairs.size()), MM); d++) 
					layerEdgeLists[i][n].push_back(distPairs[d].second);
			}
		}
		ep = selectedNeighbors[0];
	}

	if (l == layerEdgeLists.size() - 1) 
		enterNode = nid;
	
	// cout << "[*] Done Insert" << endl;
}
