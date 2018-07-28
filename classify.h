#pragma once

#include <unordered_map>

typedef dlib::matrix<float,128,1> sample_type;

int classify(
	const std::vector<sample_type>& samples,
	std::unordered_map<int, int>& cats,
	const sample_type& test_sample
);
