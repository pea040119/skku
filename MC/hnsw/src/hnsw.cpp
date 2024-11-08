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
	set<pair<double, int>> candidates;
	set<pair<double, int>> nearestNeighbors;
	unordered_set<int> isVisited;
	Item *temp;

	// #pragma omp critical 
	{
		temp = &items[ep];
	}
	

	double td = q.dist(*temp);
	// #pragma omp critical
	candidates.insert(make_pair(td, ep));
	nearestNeighbors.insert(make_pair(td, ep));
	isVisited.insert(ep);
	while (!candidates.empty()) {
		auto ci = candidates.begin(); 
		candidates.erase(candidates.begin());
		int nid = ci->second;
		auto fi = nearestNeighbors.end(); 
		fi--;
		if (ci->first > fi->first) 
			break;
		// #pragma omp parallel for parallelnum_threads(mp parallel for num_thread40) shared(layerEdgeLists, items, isVisited, candidates, nearestNeighbors)
		vector<int>* temp_layer;
		// #pragma omp critical
		{
			temp_layer = &layerEdgeLists[lc][nid];
		}
		for (int ed: *temp_layer) {
			if (isVisited.find(ed) != isVisited.end()) 
				continue;
			// cout << "check" << endl;
			fi = nearestNeighbors.end(); 
			fi--;
			isVisited.insert(ed);
			// #pragma omp critical
			{
				temp = &items[ed];
			}
			td = q.dist(*temp);
			if ((td < fi->first) || nearestNeighbors.size() < ef) {
				candidates.insert(make_pair(td, ed));
				nearestNeighbors.insert(make_pair(td, ed));
				if (nearestNeighbors.size() > ef) 
					nearestNeighbors.erase(fi);
			}
		}
	}
	vector<int> results;
	for(auto &p: nearestNeighbors) results.push_back(p.second);
	return results;
}



vector<int> HNSWGraph::KNNSearch(Item& q, int K) {
	// cout << "[*] Start KNNSearch" << endl;
	int maxLyer;
	#pragma omp critical(lock)
	{
		maxLyer = layerEdgeLists.size() - 1;
	}
	int ep = enterNode;
    vector<int> result;
    
	#pragma omp critical(lock)
	{
		for (int l = maxLyer; l >= 1; l--)
			ep = searchLayer(q, ep, 1, l)[0];
		result = searchLayer(q, ep, K, 0);
	}
    
	return result;
}


void HNSWGraph::addEdge(int st, int ed, int lc) {
	if (st == ed) return;
	// #pragma omp critical(lock)
	layerEdgeLists[lc][st].push_back(ed);
	layerEdgeLists[lc][ed].push_back(st);
}


void HNSWGraph::Insert(Item& q) {
	int nid = items.size();
	itemNum++; 
	items.push_back(q);
	
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

	int ep = enterNode;
	for (int i = maxLyer; i > l; i--) 
		ep = searchLayer(q, ep, 1, i)[0];
	
	for (int i = min(l, maxLyer); i >= 0; i--) {
		int MM = l == 0 ? MMax0 : MMax;
		vector<int> neighbors = searchLayer(q, ep, efConstruction, i);
		vector<int> selectedNeighbors = vector<int>(neighbors.begin(), neighbors.begin()+min(int(neighbors.size()), M));
		for (int n: selectedNeighbors) 
			addEdge(n, nid, i);
		#pragma omp parallel for num_threads(4) shared(layerEdgeLists, items, selectedNeighbors)
		for (int n: selectedNeighbors) {
			if (layerEdgeLists[i][n].size() > MM) {
				vector<pair<double, int>> distPairs;
				// #pragma omp parallel for num_threads(4) shared(distPairs) firstprivate(i, n, layerEdgeLists, items)
				for (int nn: layerEdgeLists[i][n]) {
					int local_dist = items[n].dist(items[nn]);
					distPairs.emplace_back(local_dist, nn);
				}
				sort(distPairs.begin(), distPairs.end());
				
				layerEdgeLists[i][n].clear();
				for (int d = 0; d < min(int(distPairs.size()), MM); d++) 
					layerEdgeLists[i][n].push_back(distPairs[d].second);
			}
		}
		ep = selectedNeighbors[0];
	}
	if (l == layerEdgeLists.size() - 1) enterNode = nid;
}
