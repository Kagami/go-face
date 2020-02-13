#include <shared_mutex>
#include <dlib/dnn.h>
#include <dlib/image_loader/image_loader.h>
#include <dlib/image_loader/jpeg_loader.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/graph_utils.h>
#include <dlib/image_io.h>
#include "facerec.h"
#include "classify.h"
#include "utils.h"

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

void FaceRec::setAge(const char* age_path) {
    deserialize(std::string(age_path)) >> age_net_;
}

std::vector<rectangle> FaceRec::detect(image_t& img) {
    std::lock_guard<std::mutex> lock(detector_mutex_);
        
    while(img.size() < min_image_size) {
        pyramid_up(img);
    }

    return detector_(img);
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

int FaceRec::gender(image_pointer *p) {
    full_object_detection shape;
    std::lock_guard<std::mutex> lock(net_mutex_);
    image_t img = *((image_t*)p->img);

    if (p->shape) {
        shape = *((full_object_detection*)p->shape);
    } else {
        rectangle rect = *((rectangle*)p->rect);
        shape = sp_(img, rect);
        p->shape = new full_object_detection(shape);
    }
    
    if (shape.num_parts()) {
        image_t face_chip;
        extract_image_chip(img, get_face_chip_details(shape, 32), face_chip);
        return int(gender_net_(face_chip));
    }
    
    return 0;
}

int FaceRec::age(image_pointer *p) {
    full_object_detection shape;
    std::lock_guard<std::mutex> lock(net_mutex_);
    image_t img = *((image_t*)p->img);

    if (p->shape) {
        shape = *((full_object_detection*)p->shape);
    } else {
        rectangle rect = *((rectangle*)p->rect);
        shape = sp_(img, rect);
        p->shape = new full_object_detection(shape);
    }
    
    snet.subnet() = age_net_.subnet();

    if (shape.num_parts()) {
        float confidence;
        image_t face_chip;
        extract_image_chip(img, get_face_chip_details(shape, 64), face_chip);
        matrix<float, 1, number_of_age_classes> p = mat(snet(face_chip));
        return int(get_estimated_age(p, confidence));
    }

    return 0;
}

std::tuple<descriptor, full_object_detection> FaceRec::recognize(image_pointer *p) {
    descriptor descr;
    full_object_detection shape;
    std::lock_guard<std::mutex> lock(net_mutex_);
    image_t img = *((image_t*)p->img);

    if (p->shape) {
        shape = *((full_object_detection*)p->shape);
    } else {
        rectangle rect = *((rectangle*)p->rect);
        shape = sp_(img, rect);
        p->shape = new full_object_detection(shape);
    }

    if (shape.num_parts()) {
        image_t face_chip;
        extract_image_chip(img, get_face_chip_details(shape, size, padding), face_chip);

        if (jittering > 0) {
            descr = mean(mat(net_(jitter_image(std::move(face_chip), jittering))));
        } else {
            descr = net_(face_chip);
        }
    }

    return std::make_tuple(descr, shape);
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

facesret* facerec_detect(facesret *ret, facerec* rec, image_t &img, int type) {
	FaceRec* cls = (FaceRec*)(rec->cls);
	std::vector<rectangle> rects;

    if (type == 0 ) {
        rects = cls->detect(img);
    } else {
        rects = cls->detectCNN(img);
    }

    ret->num_faces = rects.size();    

    if (ret->num_faces == 0)
		return ret;

	ret->rectangles = (long*)malloc(ret->num_faces * RECT_LEN * sizeof(long));
    ret->p = new image_pointer[ret->num_faces];

	for (int i = 0; i < ret->num_faces; i++) {
        ret->p[i].img = new image_t(img);
        ret->p[i].rect = new rectangle(rects[i]);
        ret->p[i].shape = 0;     
        long* dst = ret->rectangles + i * RECT_LEN;
        dst[0] = rects[i].left();
        dst[1] = rects[i].top();
        dst[2] = rects[i].right();
        dst[3] = rects[i].bottom();
    }
	return ret;
}

// Classify
void FaceRec::setSamples(std::vector<descriptor>&& samples, std::vector<int>&& cats) {
    std::unique_lock<std::shared_mutex> lock(samples_mutex_);
    samples_ = std::move(samples);
    cats_ = std::move(cats);
}

int FaceRec::classify(const descriptor& test_sample, float tolerance) {
    std::shared_lock<std::shared_mutex> lock(samples_mutex_);
    return classify_(samples_, cats_, test_sample, tolerance);
}
