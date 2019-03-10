#include <dlib/graph_utils.h>
#include "classify.h"

int classify(
	const std::vector<descriptor>& samples,
	const std::unordered_map<int, int>& cats,
	const descriptor& test_sample,
	float tolerance
) {
	if (samples.size() == 0)
		return -1;

	std::vector<std::pair<int, float>> distances;
	distances.reserve(samples.size());
	auto dist_func = dlib::squared_euclidean_distance();
	int idx = 0;
	for (const auto& sample : samples) {
		float dist = dist_func(sample, test_sample);
		if (dist < tolerance)
			continue;
		distances.push_back({idx, dist});
		idx++;
	}

	if (distances.size() == 0)
		return -1;

	std::sort(
		distances.begin(), distances.end(),
		[](const auto a, const auto b) { return a.second < b.second; }
	);

	int len = std::min((int)distances.size(), 10);
	std::unordered_map<int, std::pair<int, float>> hits_by_cat;
	for (int i = 0; i < len; i++) {
		int idx = distances[i].first;
		float dist = distances[i].second;
		auto cat = cats.find(idx);
		if (cat == cats.end())
			continue;
		int cat_idx = cat->second;
		auto hit = hits_by_cat.find(cat_idx);
		if (hit == hits_by_cat.end()) {
			// printf("1 hit for %d (%d: %f)\n", cat_idx, idx, dist);
			hits_by_cat[cat_idx] = {1, dist};
		} else {
			// printf("+1 hit for %d (%d: %f)\n", cat_idx, idx, dist);
			hits_by_cat[cat_idx].first++;
		}
	}

	auto hit = std::max_element(
		hits_by_cat.begin(), hits_by_cat.end(),
		[](const auto a, const auto b) {
			auto hits1 = a.second.first;
			auto hits2 = b.second.first;
			auto dist1 = a.second.second;
			auto dist2 = b.second.second;
			if (hits1 == hits2) return dist1 > dist2;
			return hits1 < hits2;
		}
	);
	// printf("Found cat with max hits: %d\n", hit->first); fflush(stdout);
	return hit->first;
}

float distance(
	const descriptor& sample1,
	const descriptor& sample2
) {
	auto dist_func = dlib::squared_euclidean_distance();
	return dist_func(sample1, sample2);
}
