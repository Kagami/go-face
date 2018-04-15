#include <dlib/graph_utils.h>
#include "classify.h"

int classify(
	const std::vector<sample_type>& samples,
	std::unordered_map<int, int>& cats,
	const sample_type& test_sample
) {
	std::vector<std::pair<int, double>> distances;
	distances.reserve(samples.size());
	auto dist_func = dlib::squared_euclidean_distance();
	int idx = 0;
	for (const auto& sample : samples) {
		double dist = dist_func(sample, test_sample);
		distances.push_back({idx, dist});
		idx++;
	}

	std::sort(
		distances.begin(), distances.end(),
		[](const auto a, const auto b) { return a.second < b.second; }
	);

	int len = std::min((int)distances.size(), 10);
	std::unordered_map<int, std::pair<int, double>> hits_by_cat;
	for (int i = 0; i < len; i++) {
		int idx = distances[i].first;
		double dist = distances[i].second;
		auto cat = cats.find(idx);
		if (cat == cats.end())
			continue;
		int catIdx = cat->second;
		auto hit = hits_by_cat.find(catIdx);
		if (hit == hits_by_cat.end()) {
			// printf("1 hit for %d (%d: %f)\n", catIdx, idx, dist);
			hits_by_cat[catIdx] = {1, dist};
		} else {
			// printf("+1 hit for %d (%d: %f)\n", catIdx, idx, dist);
			hits_by_cat[catIdx].first++;
		}
	}

	auto hit = std::max_element(
		hits_by_cat.begin(), hits_by_cat.end(),
		[](const auto a, const auto b) {
			int hits1 = a.second.first;
			int hits2 = b.second.first;
			double dist1 = a.second.second;
			double dist2 = b.second.second;
			if (hits1 == hits2) return dist1 > dist2;
			return hits1 < hits2;
		}
	);
	// printf("Found cat with max hits: %d\n", hit->first); fflush(stdout);
	return hit->first;
}
