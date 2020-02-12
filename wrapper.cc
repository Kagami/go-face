// Plain C interface for Go.
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

facerec* facerec_init() {
	facerec* rec = (facerec*)calloc(1, sizeof(facerec));
	try {
		FaceRec* cls = new FaceRec();
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

void facerec_config_set_size(facerec* rec, unsigned long size) {
	FaceRec* cls = (FaceRec*)(rec->cls);
	cls->setSize(size);
}

void facerec_config_set_padding(facerec* rec, double padding) {
	FaceRec* cls = (FaceRec*)(rec->cls);
	cls->setPadding(padding);
}

void facerec_config_set_jittering(facerec* rec, int jittering) {
	FaceRec* cls = (FaceRec*)(rec->cls);
	cls->setJittering(jittering);
}

void facerec_config_set_min_image_size(facerec* rec, int min_image_size) {
	FaceRec* cls = (FaceRec*)(rec->cls);
	cls->setMinImageSize(min_image_size);
}

void facerec_set_cnn(facerec* rec, const char *file) {
	FaceRec* cls = (FaceRec*)(rec->cls);
	cls->setCNN(file);
}

void facerec_set_shape(facerec* rec, const char *file) {
	FaceRec* cls = (FaceRec*)(rec->cls);
	cls->setShape(file);
}

void facerec_set_descriptor(facerec* rec, const char *file) {
	FaceRec* cls = (FaceRec*)(rec->cls);
	cls->setDescriptor(file);
}

void facerec_set_gender(facerec* rec, const char *file) {
	FaceRec* cls = (FaceRec*)(rec->cls);
	cls->setGender(file);
}

void facerec_set_age(facerec* rec, const char *file) {
	FaceRec* cls = (FaceRec*)(rec->cls);
	cls->setAge(file);
}

facesret* facerec_detect_from_file(facerec* rec, image_pointer *p, const char* file,int type) {
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
    
    return facerec_detect(ret, p, rec, img, type);
}

facesret* facerec_detect_from_buffer(facerec* rec, image_pointer *p, unsigned char* img_data, int len,int type) {
	facesret* ret = (facesret*)calloc(1, sizeof(facesret));
	image_t img;
	std::vector<rectangle> rects;

	try {
		// TODO(Kagami): Support more file types?
		load_jpeg(img, img_data, size_t(len));
	} catch(image_load_error& e) {
		ret->err_str = strdup(e.what());
		ret->err_code = IMAGE_LOAD_ERROR;
		return ret;
	} catch (std::exception& e) {
		ret->err_str = strdup(e.what());
		ret->err_code = UNKNOWN_ERROR;
		return ret;
	}
    
    return facerec_detect(ret, p, rec, img, type);
}

faceret* facerec_recognize(facerec* rec, image_pointer *p, int x, int y, int x1, int y1) {
    faceret* ret = (faceret*)calloc(1, sizeof(faceret));
	FaceRec* cls = (FaceRec*)(rec->cls);
    image_t img = *((image_t*)p->p);

	descriptor descr;
	full_object_detection shape;
	rectangle r = rectangle(x,y,x1,y1);

   try {
		std::tie(descr, shape) = cls->recognize(img, r);
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

int facerec_get_gender(facerec* rec, image_pointer *p, int x, int y, int x1, int y1) {
	FaceRec* cls = (FaceRec*)(rec->cls);
    image_t img = *((image_t*)p->p);
	rectangle r = rectangle(x,y,x1,y1);
    
	return cls->gender(img, r);
}

int facerec_get_age(facerec* rec, image_pointer *p, int x, int y, int x1, int y1) {
	FaceRec* cls = (FaceRec*)(rec->cls);
    image_t img = *((image_t*)p->p);
	rectangle r = rectangle(x,y,x1,y1);
    
	return cls->age(img, r);
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
	cls->setSamples(std::move(samples), std::move(cats));
}

int facerec_classify(facerec* rec, const float* c_test_sample, float tolerance) {
	FaceRec* cls = (FaceRec*)(rec->cls);
	descriptor test_sample = mat(c_test_sample, DESCR_LEN, 1);
	return cls->classify(test_sample, tolerance);
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

void image_pointer_free(image_pointer* p) {
	if (p) {
		if (p->p) {
			image_t *img = ((image_t*)p->p);
			delete img;
			p->p = NULL;
		}
	}
}