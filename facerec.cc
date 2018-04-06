#include <dlib/dnn.h>
#include <dlib/image_loader/image_loader.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include "facerec.h"
#include "jpeg_mem_loader.h"
#include "classify.h"

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

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

typedef matrix<float,0,1> descriptor;
typedef std::pair<std::vector<rectangle>, std::vector<descriptor>> faces;

static const size_t RECT_LEN = 4;
static const size_t DESCR_LEN = 128;
static const size_t RECT_SIZE = RECT_LEN * sizeof(long);
static const size_t DESCR_SIZE = DESCR_LEN * sizeof(float);

class FaceRec {
public:
	FaceRec(const char* model_dir) {
		detector_ = get_frontal_face_detector();

		std::string dir = model_dir;
		std::string shape_predictor_path = dir + "/shape_predictor_5_face_landmarks.dat";
		std::string resnet_path = dir + "/dlib_face_recognition_resnet_model_v1.dat";

		deserialize(shape_predictor_path) >> sp_;
		deserialize(resnet_path) >> net_;
	}

	// TODO(Kagami): Jittering?
	faces Recognize(matrix<rgb_pixel>& img, int max_faces) {
		std::vector<rectangle> rects = detector_(img);
		std::vector<descriptor> descrs;

		// Short circuit.
		if (rects.size() == 0 || (max_faces && rects.size() > max_faces))
			return {rects, descrs};

		std::vector<matrix<rgb_pixel>> face_imgs;
		for (auto rect : rects) {
			auto shape = sp_(img, rect);
			matrix<rgb_pixel> face_chip;
			extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
			face_imgs.push_back(std::move(face_chip));
		}

		descrs = net_(face_imgs);
		return {rects, descrs};
	}
private:
	frontal_face_detector detector_;
	shape_predictor sp_;
	anet_type net_;
};

// Plain C interface for Go.

facerec* facerec_init(const char* model_dir) {
	facerec* rec = (facerec*)calloc(1, sizeof(facerec));
	try {
		FaceRec* cls = new FaceRec(model_dir);
		rec->cls = (void*)cls;
	} catch(serialization_error& e) {
		rec->err_str = strdup(e.what());
		rec->err_code = SERIALIZATION_ERROR;
	} catch (std::exception& e) {
		rec->err_str = strdup(e.what());
		rec->err_code = UNKNOWN_ERROR;
	}
	return rec;
}

faceret* facerec_recognize(facerec* rec, const uint8_t* img_data, int len, int max_faces) {
	faceret* ret = (faceret*)calloc(1, sizeof(faceret));
	FaceRec* cls = (FaceRec*)(rec->cls);
	matrix<rgb_pixel> img;
	try {
		// TODO(Kagami): Support more file types?
		load_mem_jpeg(img, img_data, len);
		faces faces = cls->Recognize(img, max_faces);
		std::vector<rectangle> rects = faces.first;
		std::vector<descriptor> descrs = faces.second;
		ret->num_faces = descrs.size();
		if (ret->num_faces == 0)
			return ret;
		ret->rectangles = (long*)malloc(ret->num_faces * RECT_SIZE);
		for (int i = 0; i < ret->num_faces; i++) {
			long* dst = ret->rectangles + i * 4;
			dst[0] = rects[i].left();
			dst[1] = rects[i].top();
			dst[2] = rects[i].right();
			dst[3] = rects[i].bottom();
		}
		ret->descriptors = (float*)malloc(ret->num_faces * DESCR_SIZE);
		for (int i = 0; i < ret->num_faces; i++) {
			void* dst = (uint8_t*)(ret->descriptors) + i * DESCR_SIZE;
			void* src = (void*)&descrs[i](0,0);
			memcpy(dst, src, DESCR_SIZE);
		}
	} catch(image_load_error& e) {
		ret->err_str = strdup(e.what());
		ret->err_code = IMAGE_LOAD_ERROR;
	} catch (std::exception& e) {
		ret->err_str = strdup(e.what());
		ret->err_code = UNKNOWN_ERROR;
	}
	return ret;
}

int facerec_classify(facerec* rec, const float* descriptors, int len, const float* descriptor) {
	std::vector<sample_type> samples;
	for (int i = 0; i < len; i++) {
		sample_type sample = mat(descriptors + i*DESCR_LEN, DESCR_LEN, 1);
		samples.push_back(sample);
	}
	sample_type test_sample = mat(descriptor, DESCR_LEN, 1);
	return classify(samples, test_sample);
}

void facerec_free(facerec* rec) {
	if (rec) {
		if (rec->cls) {
			FaceRec* cls = (FaceRec*)(rec->cls);
			delete cls;
			rec->cls = NULL;
		}
		free(rec);
	}
}
