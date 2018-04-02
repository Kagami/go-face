#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include "facerec.h"

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
static const size_t DESCR_SIZE = 128 * sizeof(float);

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
	std::vector<descriptor> Recognize(const char* img_path) {
		matrix<rgb_pixel> img;
		load_image(img, img_path);

		std::vector<matrix<rgb_pixel>> faces;
		for (auto face : detector_(img)) {
			auto shape = sp_(img, face);
			matrix<rgb_pixel> face_chip;
			extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
			faces.push_back(std::move(face_chip));
		}

		// Short circuit.
		if (faces.size() == 0)
			return {};

		std::vector<descriptor> descriptors = net_(faces);
		return descriptors;
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

faceret* facerec_recognize(facerec* rec, const char* img_path) {
	faceret* ret = (faceret*)calloc(1, sizeof(faceret));
	FaceRec* cls = (FaceRec*)(rec->cls);
	try {
		std::vector<descriptor> descriptors = cls->Recognize(img_path);
		ret->num_faces = descriptors.size();
		if (ret->num_faces == 0)
			return ret;
		ret->descriptors = (float*)malloc(ret->num_faces * DESCR_SIZE);
		for (int i = 0; i < ret->num_faces; i++) {
			void* dst = (uint8_t*)(ret->descriptors) + i * DESCR_SIZE;
			void* src = (void*)&descriptors[i](0,0);
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
