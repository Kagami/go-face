#include <dlib/graph_utils.h>
#include "classify.h"

int classify(std::vector<sample_type>& samples, sample_type& test_sample) {
	auto dist_func = dlib::squared_euclidean_distance();
	double min_distance = INFINITY;
	int min_label = -1;
	int i = 0;
	for (auto sample : samples) {
		auto dist = dist_func(sample, test_sample);
		if (dist < min_distance) {
			min_distance = dist;
			min_label = i;
		}
		i++;
	}
	return min_label;
}
