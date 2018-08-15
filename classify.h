#pragma once

#include <unordered_map>

typedef dlib::matrix<float,0,1> descriptor;

int classify(
	const std::vector<descriptor>& samples,
	const std::unordered_map<int, int>& cats,
	const descriptor& test_sample
);
