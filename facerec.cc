#include <shared_mutex>
#include <dlib/dnn.h>
#include <dlib/image_loader/image_loader.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/graph_utils.h>
#include <dlib/image_io.h>
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

std::vector<rectangle> FaceRec::Detect(image_t& img) {
    	std::vector<rectangle> rects;
    	std::lock_guard<std::mutex> lock(detector_mutex_);
        
    while(img.size() < min_image_size) {
        pyramid_up(img);
    }
        
    rects = detector_(img);
    return rects;
}

std::vector<rectangle> FaceRec::DetectCNN(image_t& img) {
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

std::tuple<descriptor, full_object_detection> FaceRec::Recognize(const image_t& img,rectangle rect) {
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

void FaceRec::SetSamples(std::vector<descriptor>&& samples, std::vector<int>&& cats) {
    std::unique_lock<std::shared_mutex> lock(samples_mutex_);
    samples_ = std::move(samples);
    cats_ = std::move(cats);
}

int FaceRec::Classify(const descriptor& test_sample, float tolerance) {
    std::shared_lock<std::shared_mutex> lock(samples_mutex_);
    return classify(samples_, cats_, test_sample, tolerance);
}

void FaceRec::Config(unsigned long new_size, double new_padding, int new_jittering, int new_min_image_size) {
    size = new_size;
    padding = new_padding;
    jittering = new_jittering;
    min_image_size = new_min_image_size;
}
    
// Plain C interface for Go.

facerec* facerec_init(const char* resnet_path, const char* cnn_resnet_path, const char* shape_predictor_path) {
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
void facerec_config(facerec* rec, unsigned long size, double padding, int jittering, int min_image_size) {
	FaceRec* cls = (FaceRec*)(rec->cls);
	cls->Config(size,padding, jittering, min_image_size);
}

facesret* facerec_detect_file(facerec* rec, const char* file,int type) {
	facesret* ret = (facesret*)calloc(1, sizeof(facesret));
	image_t img;
	std::vector<rectangle> rects;

	try {
		// TODO(Kagami): Support more file types?
        // Danil_e71: support png, gif, bmp, jpg from file
        load_image(img,file);
	} catch(image_load_error& e) {
		ret->err_str = strdup(e.what());
		ret->err_code = IMAGE_LOAD_ERROR;
		return ret;
	} catch (std::exception& e) {
		ret->err_str = strdup(e.what());
		ret->err_code = UNKNOWN_ERROR;
		return ret;
	}
    
    return facerec_detect(ret, rec, img, type);
}

facesret* facerec_detect_buffer(facerec* rec, const uint8_t* img_data, int len,int type) {
	facesret* ret = (facesret*)calloc(1, sizeof(facesret));
	image_t img;
	std::vector<rectangle> rects;

	try {
		// TODO(Kagami): Support more file types?
		load_mem_jpeg(img, img_data, len);
	} catch(image_load_error& e) {
		ret->err_str = strdup(e.what());
		ret->err_code = IMAGE_LOAD_ERROR;
		return ret;
	} catch (std::exception& e) {
		ret->err_str = strdup(e.what());
		ret->err_code = UNKNOWN_ERROR;
		return ret;
	}
    
    return facerec_detect(ret, rec, img, type);
}

facesret* facerec_detect(facesret* ret, facerec* rec, image_t img, int type) {
	FaceRec* cls = (FaceRec*)(rec->cls);
	std::vector<rectangle> rects;
    
    if (type == 0 ) {
        rects = cls->Detect(img);
    } else {
        rects = cls->DetectCNN(img);
    }
    
    ret->num_faces = rects.size();
    ret->img = new image_pointer{&img};
	
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

faceret* facerec_recognize(facerec* rec, image_pointer *pointer, int x, int y, int x1, int y1) {
    	faceret* ret = (faceret*)calloc(1, sizeof(faceret));
	FaceRec* cls = (FaceRec*)(rec->cls);
	descriptor descr;
	full_object_detection shape;
    
	try {
		std::tie(descr, shape) = cls->Recognize(*(pointer->img), rectangle(x,y,x1,y1));
	} catch(image_load_error& e) {
		ret->err_str = strdup(e.what());
		ret->err_code = IMAGE_LOAD_ERROR;
		return ret;
	} catch (std::exception& e) {
		ret->err_str = strdup(e.what());
		ret->err_code = UNKNOWN_ERROR;
		return ret;
	}

	ret->descriptor = (float*)malloc(DESCR_LEN * sizeof(float));
	memcpy((uint8_t*)(ret->descriptor), (void*)&descr(0,0), DESCR_LEN * sizeof(float));
	
	ret->num_shape = shape.num_parts();
	ret->shape = (long*)malloc(ret->num_shape * SHAPE_LEN * sizeof(long));

	long* dst = ret->shape;
	for (int j = 0; j < ret->num_shape; j++) {
		dst[j*SHAPE_LEN] = shape.part(j).x();
		dst[j*SHAPE_LEN+1] = shape.part(j).y();
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

static std::vector<image_t> jitter_image(
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
