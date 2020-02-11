#pragma once
#ifndef FACEREC_H
#define FACEREC_H

#ifdef __cplusplus
#include <shared_mutex>
#include <dlib/dnn.h>
#include <dlib/image_loader/image_loader.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/graph_utils.h>
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
	std::vector<descriptor> samples_;
	std::vector<int> cats_;
	int jittering = 0;
	unsigned long size = 150;
	double padding = 0.25;
    int min_image_size;
    
public:
	FaceRec(const char*, const char*, const char*);
    std::vector<rectangle> Detect(image_t&);
    std::vector<rectangle> DetectCNN(image_t&);
    std::tuple<descriptor, full_object_detection> Recognize(const image_t&,rectangle);
    
	void SetSamples(std::vector<descriptor>&&, std::vector<int>&&);
	int Classify(const descriptor&, float);
    void Config(unsigned long, double, int, int);
};

static std::vector<image_t> jitter_image(const image_t& img,int count);

#else
typedef void *image_t;
#endif


#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	IMAGE_LOAD_ERROR,
	SERIALIZATION_ERROR,
	UNKNOWN_ERROR,
} err_code;

typedef struct facerec {
	void* cls;
	const char* err_str;
	err_code err_code;
	int jittering;
	unsigned long size;
	double padding;
} facerec;

typedef struct image_pointer {
    image_t *img;
} image_pointer;

typedef struct facesret {
    image_pointer *img;
	int num_faces;
	long* rectangles;
	const char* err_str;
	err_code err_code;
} facesret;

typedef struct faceret {
	float *descriptor;
	int num_shape;
	long* shape;
	const char* err_str;
	err_code err_code;
} faceret;

#define RECT_LEN   4
#define DESCR_LEN  128
#define SHAPE_LEN  2

facerec* facerec_init(const char*,const char*,const char*);
facesret* facerec_detect_file(facerec*, const char*,int);
facesret* facerec_detect_buffer(facerec*, const uint8_t*, int, int);
facesret* facerec_detect(facesret*, facerec*, image_t, int);

faceret* facerec_recognize(facerec*, image_pointer*, int, int, int, int);
void facerec_set_samples(facerec*, const float*, const int32_t*, int);
int facerec_classify(facerec*, const float*, float);
void facerec_free(facerec*);
void facerec_config(facerec*, unsigned long, double, int, int);



#ifdef __cplusplus
}
#endif
#endif /* FACEREC_H */
