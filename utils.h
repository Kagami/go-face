#pragma once
#ifndef UTILS_H
#define UTILS_H

#ifdef __cplusplus
#include <dlib/graph_utils.h>

std::vector<image_t> jitter_image(const image_t& img,int count);
uint8_t get_estimated_age(matrix<float, 1, number_of_age_classes>&, float&);

#endif
#endif /* UTILS_H */
