#include <shared_mutex>
#include <dlib/dnn.h>
#include <dlib/image_loader/image_loader.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/graph_utils.h>
#include "facerec.h"
#include "jpeg_mem_loader.h"
#include "classify.h"

using namespace dlib;

FaceRec::FaceRec(const char* resnet_path,const char* cnn_resnet_path,const char* shape_predictor_path) {
		detector_ = get_frontal_face_detector();

		deserialize(std::string(shape_predictor_path)) >> sp_;
		deserialize(std::string(resnet_path)) >> net_;
		deserialize(std::string(cnn_resnet_path)) >> cnn_net_;
	}

std::tuple<std::vector<rectangle>, std::vector<descriptor>, std::vector<full_object_detection>>
	FaceRec::Recognize(const matrix<rgb_pixel>& img,int max_faces,int type) {
		std::vector<rectangle> rects;
		std::vector<descriptor> descrs;
		std::vector<full_object_detection> shapes;

		if(type == 0) {
			std::lock_guard<std::mutex> lock(detector_mutex_);
			rects = detector_(img);
		} else{
			std::lock_guard<std::mutex> lock(cnn_net_mutex_);
			auto dets = cnn_net_(img);
            for (auto&& d : dets) {
                rects.push_back(d.rect);
            }
		}

		// Short circuit.
		if (rects.size() == 0 || (max_faces > 0 && rects.size() > (size_t)max_faces))
			return {std::move(rects), std::move(descrs), std::move(shapes)};

		std::sort(rects.begin(), rects.end());

		for (const auto& rect : rects) {
			auto shape = sp_(img, rect);
			shapes.push_back(shape);
			matrix<rgb_pixel> face_chip;
			extract_image_chip(img, get_face_chip_details(shape, size, padding), face_chip);
			std::lock_guard<std::mutex> lock(net_mutex_);
			if (jittering > 0) {
				descrs.push_back(mean(mat(net_(jitter_image(std::move(face_chip), jittering)))));
			} else {
				descrs.push_back(net_(face_chip));
			}
		}

		return {std::move(rects), std::move(descrs), std::move(shapes)};
	}

	void FaceRec::SetSamples(std::vector<descriptor>&& samples, std::vector<int>&& cats) {
		std::unique_lock<std::shared_mutex> lock(samples_mutex_);
		samples_ = std::move(samples);
		cats_ = std::move(cats);
	}

	int FaceRec::Classify(const descriptor& test_sample, float tolerance) {
		std::shared_lock<std::shared_mutex> lock(samples_mutex_);
		return classify(samples_, cats_, test_sample, tolerance);
	}

    void FaceRec::Config(unsigned long new_size, double new_padding, int new_jittering) {
        size = new_size;
        padding = new_padding;
        jittering = new_jittering;
    }
    
// Plain C interface for Go.

facerec* facerec_init(const char* resnet_path,const char* cnn_resnet_path,const char* shape_predictor_path) {
	facerec* rec = (facerec*)calloc(1, sizeof(facerec));
	try {
		FaceRec* cls = new FaceRec(resnet_path,cnn_resnet_path,shape_predictor_path);
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
void facerec_config(facerec* rec, unsigned long size, double padding, int jittering) {
	FaceRec* cls = (FaceRec*)(rec->cls);
	cls->Config(size,padding,jittering);
}

faceret* facerec_recognize(facerec* rec, const uint8_t* img_data, int len, int max_faces,int type) {
	faceret* ret = (faceret*)calloc(1, sizeof(faceret));
	FaceRec* cls = (FaceRec*)(rec->cls);
	matrix<rgb_pixel> img;
	std::vector<rectangle> rects;
	std::vector<descriptor> descrs;
	std::vector<full_object_detection> shapes;

	try {
		// TODO(Kagami): Support more file types?
		load_mem_jpeg(img, img_data, len);
		std::tie(rects, descrs, shapes) = cls->Recognize(img, max_faces,type);
	} catch(image_load_error& e) {
		ret->err_str = strdup(e.what());
		ret->err_code = IMAGE_LOAD_ERROR;
		return ret;
	} catch (std::exception& e) {
		ret->err_str = strdup(e.what());
		ret->err_code = UNKNOWN_ERROR;
		return ret;
	}
	ret->num_faces = descrs.size();

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
	ret->descriptors = (float*)malloc(ret->num_faces * DESCR_LEN * sizeof(float));
	for (int i = 0; i < ret->num_faces; i++) {
		void* dst = (uint8_t*)(ret->descriptors) + i * DESCR_LEN * sizeof(float);
		void* src = (void*)&descrs[i](0,0);
		memcpy(dst, src, DESCR_LEN * sizeof(float));
	}
	ret->num_shapes = shapes[0].num_parts();
	ret->shapes = (long*)malloc(ret->num_faces * ret->num_shapes * SHAPE_LEN * sizeof(long));
	for (int i = 0; i < ret->num_faces; i++) {
		long* dst = ret->shapes + i * ret->num_shapes * SHAPE_LEN;
		const auto& shape = shapes[i];
		for (int j = 0; j < ret->num_shapes; j++) {
			dst[j*SHAPE_LEN] = shape.part(j).x();
			dst[j*SHAPE_LEN+1] = shape.part(j).y();
		}
	}
	return ret;
}

void facerec_set_samples(
	facerec* rec,
	const float* c_samples,
	const int32_t* c_cats,
	int len
) {
	FaceRec* cls = (FaceRec*)(rec->cls);
	std::vector<descriptor> samples;
	samples.reserve(len);
	for (int i = 0; i < len; i++) {
		descriptor sample = mat(c_samples + i*DESCR_LEN, DESCR_LEN, 1);
		samples.push_back(std::move(sample));
	}
	std::vector<int> cats(c_cats, c_cats + len);
	cls->SetSamples(std::move(samples), std::move(cats));
}

int facerec_classify(facerec* rec, const float* c_test_sample, float tolerance) {
	FaceRec* cls = (FaceRec*)(rec->cls);
	descriptor test_sample = mat(c_test_sample, DESCR_LEN, 1);
	return cls->Classify(test_sample, tolerance);
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

static std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img,
    int count
)
{
    // All this function does is make count copies of img, all slightly jittered by being
    // zoomed, rotated, and translated a little bit differently. They are also randomly
    // mirrored left to right.
    thread_local dlib::rand rnd;

    std::vector<matrix<rgb_pixel>> crops;
    for (int i = 0; i < count; ++i)
        crops.push_back(jitter_image(img,rnd));

    return crops;
}
