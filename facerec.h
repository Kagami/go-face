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


// don't kill me))
using namespace dlib;


template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

using cnn_anet_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

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
    
public:
	FaceRec(const char* resnet_path,const char* cnn_resnet_path,const char* shape_predictor_path);
	std::tuple<std::vector<rectangle>, std::vector<descriptor>, std::vector<full_object_detection>>
	Recognize(const matrix<rgb_pixel>& img,int max_faces,int type);
	void SetSamples(std::vector<descriptor>&& samples, std::vector<int>&& cats);
	int Classify(const descriptor& test_sample, float tolerance);
    void Config(unsigned long new_size, double new_padding, int new_jittering);
};

static std::vector<matrix<rgb_pixel>> jitter_image(const matrix<rgb_pixel>& img,int count);

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

typedef struct faceret {
	int num_faces;
	long* rectangles;
	float* descriptors;
	int num_shapes;
	long* shapes;
	const char* err_str;
	err_code err_code;
} faceret;

#define RECT_LEN   4
#define DESCR_LEN  128
#define SHAPE_LEN  2

facerec* facerec_init(const char* resnet_path,const char* cnn_resnet_path,const char* shape_predictor_path);
faceret* facerec_recognize(facerec* rec, const uint8_t* img_data, int len, int max_faces,int type);
void facerec_set_samples(facerec* rec, const float* descriptors, const int32_t* cats, int len);
int facerec_classify(facerec* rec, const float* descriptor, float tolerance);
void facerec_free(facerec* rec);
void facerec_config(facerec* rec, unsigned long size, double padding, int jittering);



#ifdef __cplusplus
}
#endif
#endif /* FACEREC_H */
