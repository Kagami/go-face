#pragma once
#ifndef CLASSIFY_H
#define CLASSIFY_H

typedef dlib::matrix<float,0,1> descriptor;

int classify(
	const std::vector<descriptor>& samples,
	const std::vector<int>& cats,
	const descriptor& test_sample,
	float tolerance
);
#endif /* CLASSIFY_H */
