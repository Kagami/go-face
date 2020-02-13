#include <dlib/graph_utils.h>
#include <dlib/image_io.h>
#include "facerec.h"
#include "utils.h"

using namespace dlib;

std::vector<image_t> jitter_image(
    const image_t& img,
    int count
)
{
    // All this function does is make count copies of img, all slightly jittered by being
    // zoomed, rotated, and translated a little bit differently. They are also randomly
    // mirrored left to right.
    thread_local dlib::rand rnd;

    std::vector<image_t> crops;
    for (int i = 0; i < count; ++i)
        crops.push_back(jitter_image(img,rnd));

    return crops;
}

// Helper function to estimage the age
uint8_t get_estimated_age(matrix<float, 1, number_of_age_classes>& p, float& confidence)
{
	float estimated_age = (0.25f * p(0));
	confidence = p(0);

	for (uint16_t i = 1; i < number_of_age_classes; i++) {
		estimated_age += (static_cast<float>(i) * p(i));
		if (p(i) > confidence) confidence = p(i);
	}

	return std::lround(estimated_age);
}
