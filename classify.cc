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
		distances.push_back(std::move(std::make_pair(idx, dist)));
		idx++;
	}

	std::sort(
		distances.begin(), distances.end(),
		[](const auto a, const auto b) -> bool { return a.second < b.second; }
	);

	int len = std::min((int)distances.size(), 10);
	std::map<int, int> hits_by_cat;
	for (int i = 0; i < len; i++) {
		int idx = distances[i].first;
		auto cat = cats.find(idx);
		if (cat == cats.end())
			continue;
		auto hits = hits_by_cat.find(cat->second);
		if (hits == hits_by_cat.end()) {
			// printf("1 hit for %d (%d: %f)\n", cat->second, idx, distances[i].second);
			hits_by_cat[cat->second] = 1;
		} else {
			// printf("%d hits for %d (%d: %f)\n", hits->second + 1, cat->second, idx, distances[i].second);
			hits_by_cat[cat->second] = hits->second + 1;
		}
	}

	auto hits = std::max_element(
		hits_by_cat.begin(), hits_by_cat.end(),
		[](const auto a, const auto b) { return a.second < b.second; }
	);
	// printf("Found cat with max hits: %d\n", hits->first);
	// fflush(stdout);
	return hits->first;
}
