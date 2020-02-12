#include <shared_mutex>
#include <dlib/dnn.h>
#include <dlib/image_loader/image_loader.h>
#include <dlib/image_loader/jpeg_loader.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/graph_utils.h>
#include <dlib/image_io.h>
#include "facerec.h"
#include "classify.h"

using namespace dlib;

FaceRec::FaceRec() {
		detector_ = get_frontal_face_detector();
}

void FaceRec::setCNN(const char* cnn_resnet_path) {
    deserialize(std::string(cnn_resnet_path)) >> cnn_net_;
}

void FaceRec::setShape(const char* shape_predictor_path) {
    deserialize(std::string(shape_predictor_path)) >> sp_;
}

void FaceRec::setDescriptor(const char* resnet_path) {
    deserialize(std::string(resnet_path)) >> net_;
}

void FaceRec::setGender(const char* gender_path) {
    deserialize(std::string(gender_path)) >> gender_net_;
}

std::vector<rectangle> FaceRec::detect(image_t& img) {
    	std::vector<rectangle> rects;
    	std::lock_guard<std::mutex> lock(detector_mutex_);
        
    while(img.size() < min_image_size) {
        pyramid_up(img);
    }
        
    rects = detector_(img);
    return rects;
}

std::vector<rectangle> FaceRec::detectCNN(image_t& img) {
    std::vector<rectangle> rects;
    	std::lock_guard<std::mutex> lock(cnn_net_mutex_);
        
    while(img.size() < min_image_size) {
        pyramid_up(img);
    }
        
	auto dets = cnn_net_(img);
    for (auto&& d : dets) {
        rects.push_back(d.rect);
    }
    return rects; 
}

int FaceRec::gender(image_t& img,rectangle rect) {
    image_t face_chip;

    std::lock_guard<std::mutex> lock(gender_net_mutex_);

    auto shape = sp_(img, rect);
        
    extract_image_chip(img, get_face_chip_details(shape, 32), face_chip);
        
    return int(gender_net_(face_chip));
}

std::tuple<descriptor, full_object_detection> FaceRec::recognize(image_t& img,rectangle rect) {
    full_object_detection shape;
    descriptor descr;
    image_t face_chip;
        
    std::lock_guard<std::mutex> lock(net_mutex_);
        
    shape = sp_(img, rect);
        
    extract_image_chip(img, get_face_chip_details(shape, size, padding), face_chip);
        
    if (jittering > 0) {
        descr = mean(mat(net_(jitter_image(std::move(face_chip), jittering))));
    } else {
        descr = net_(face_chip);
    }
        
    return std::make_tuple(descr, shape);
}

void FaceRec::setSamples(std::vector<descriptor>&& samples, std::vector<int>&& cats) {
    std::unique_lock<std::shared_mutex> lock(samples_mutex_);
    samples_ = std::move(samples);
    cats_ = std::move(cats);
}

int FaceRec::classify(const descriptor& test_sample, float tolerance) {
    std::shared_lock<std::shared_mutex> lock(samples_mutex_);
    return classify_(samples_, cats_, test_sample, tolerance);
}

void FaceRec::setSize(unsigned long new_size) {
    size = new_size;
}

void FaceRec::setPadding(double new_padding) {
    padding = new_padding;
}

void FaceRec::setJittering(int new_jittering) {
    jittering = new_jittering;
}

void FaceRec::setMinImageSize(int new_min_image_size) {
    min_image_size = new_min_image_size;
}

facesret* facerec_detect(facesret* ret, image_pointer *p, facerec* rec, image_t &img, int type) {
	FaceRec* cls = (FaceRec*)(rec->cls);
	std::vector<rectangle> rects;

    if (type == 0 ) {
        rects = cls->detect(img);
    } else {
        rects = cls->detectCNN(img);
    }

    ret->num_faces = rects.size();
    p->p = new image_t(img);

    if (ret->num_faces == 0)
		return ret;

	ret->rectangles = (long*)malloc(ret->num_faces * RECT_LEN * sizeof(long));
	for (int i = 0; i < ret->num_faces; i++) {
		long* dst = ret->rectangles + i * RECT_LEN;
		dst[0] = rects[i].left();
		dst[1] = rects[i].top();
		dst[2] = rects[i].right();
		dst[3] = rects[i].bottom();
	}
	return ret;
}

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
