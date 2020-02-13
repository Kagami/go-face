#pragma once
#ifndef FACEREC_H
#define FACEREC_H

#ifdef __cplusplus
#include <shared_mutex>
#include <dlib/dnn.h>
#include <dlib/image_loader/image_loader.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/graph_utils.h>
#include "wrapper.h"
#include "classify.h"
#include "template.h"

typedef matrix<rgb_pixel> image_t;

class FaceRec {   
private:
	std::mutex detector_mutex_;
	std::mutex net_mutex_;
	std::mutex cnn_net_mutex_;
    
	std::shared_mutex samples_mutex_;
    
	dlib::frontal_face_detector detector_;
	dlib::shape_predictor sp_;
    
	anet_type net_;
	cnn_anet_type cnn_net_;
    agender_type gender_net_;
    apredictor_t age_net_;
    
    softmax<apredictor_t::subnet_type> snet;
    
	std::vector<descriptor> samples_;
	std::vector<int> cats_;
	int jittering = 0;
	unsigned long size = 150;
	double padding = 0.25;
    int min_image_size = 0;
    
public:
	FaceRec();
    
    void setCNN(const char*);
    void setShape(const char*);
    void setDescriptor(const char*);
    void setGender(const char*);
    void setAge(const char* age_path);
    
    void setSize(unsigned long);
    void setPadding(double);
    void setJittering(int);
    void setMinImageSize(int);

    std::vector<rectangle> detect(image_t&);
    std::vector<rectangle> detectCNN(image_t&);
    
    std::tuple<descriptor, full_object_detection> recognize(image_pointer *);
    
    int gender(image_pointer *);
    int age(image_pointer *);
    
	void setSamples(std::vector<descriptor>&&, std::vector<int>&&);
	int classify(const descriptor&, float);
};

facesret* facerec_detect(facesret*, facerec*,  image_t&, int);

#else
typedef void *image_t;
#endif
#endif /* FACEREC_H */
