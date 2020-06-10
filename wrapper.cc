// Plain C interface for Go.
#include <shared_mutex>
#include <dlib/dnn.h>
#include <dlib/image_loader/image_loader.h>
#include <dlib/image_loader/jpeg_loader.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/graph_utils.h>
#include <dlib/image_io.h>
#include "facerec.h"
#include "tracker.h"
#include "classify.h"

using namespace dlib;

tracker* tracker_init() {
	tracker* tr = (tracker*)calloc(1, sizeof(tracker));
	try {
		Tracker* cls = new Tracker();
		tr->cls = (void*)cls;
	} catch(serialization_error& e) {
		tr->err_str = strdup(e.what());
		tr->err_code = SERIALIZATION_ERROR;
	} catch (std::exception& e) {
		tr->err_str = strdup(e.what());
		tr->err_code = UNKNOWN_ERROR;
	}
	return tr;
}

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

void facerec_set_custom(facerec* rec, const char *file) {
	FaceRec* cls = (FaceRec*)(rec->cls);
	cls->setCustom(file);
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

facesret* start_track_from_file(tracker* tr, const char* file,int x1, int y1, int x2, int y2) {
    Tracker* cls = (Tracker*)(tr->cls);
	facesret* ret = (facesret*)calloc(1, sizeof(facesret));
	image_t img;
    
    	try {
		// TODO(Kagami): Support more file types?
        // Danil_e71: support png, gif, bmp, jpg from file
        load_image(img,file);
        cls->StartTrack(img,rectangle(x1,y1,x2,y2));
	} catch(image_load_error& e) {
		ret->err_str = strdup(e.what());
		ret->err_code = IMAGE_LOAD_ERROR;
		return ret;
	} catch (std::exception& e) {
		ret->err_str = strdup(e.what());
		ret->err_code = UNKNOWN_ERROR;
		return ret;
	}

    return ret;
}

facesret* start_track_from_buffer(tracker* tr, unsigned char* img_data, int len,int x1, int y1, int x2, int y2) {
    Tracker* cls = (Tracker*)(tr->cls);
	facesret* ret = (facesret*)calloc(1, sizeof(facesret));
	image_t img;

    	try {
		// TODO(Kagami): Support more file types?
		load_jpeg(img, img_data, size_t(len));
        cls->StartTrack(img,rectangle(x1,y1,x2,y2));
	} catch(image_load_error& e) {
		ret->err_str = strdup(e.what());
		ret->err_code = IMAGE_LOAD_ERROR;
		return ret;
	} catch (std::exception& e) {
		ret->err_str = strdup(e.what());
		ret->err_code = UNKNOWN_ERROR;
		return ret;
	}

    return ret;
}

update_ret* update_track_from_file(tracker* tr, const char* file) {
    Tracker* cls = (Tracker*)(tr->cls);
	update_ret* ret = (update_ret*)calloc(1, sizeof(update_ret));
	image_t img;
    
    	try {
		// TODO(Kagami): Support more file types?
        // Danil_e71: support png, gif, bmp, jpg from file
        load_image(img,file);
        ret->confidence = cls->Update(img);
	} catch(image_load_error& e) {
		ret->err_str = strdup(e.what());
		ret->err_code = IMAGE_LOAD_ERROR;
		return ret;
	} catch (std::exception& e) {
		ret->err_str = strdup(e.what());
		ret->err_code = UNKNOWN_ERROR;
		return ret;
	}

    return ret;
}

update_ret* update_track_from_buffer(tracker* tr, unsigned char* img_data, int len) {
    Tracker* cls = (Tracker*)(tr->cls);
	update_ret* ret = (update_ret*)calloc(1, sizeof(update_ret));
	image_t img;

    	try {
		// TODO(Kagami): Support more file types?
		load_jpeg(img, img_data, size_t(len));
        ret->confidence = cls->Update(img);
	} catch(image_load_error& e) {
		ret->err_str = strdup(e.what());
		ret->err_code = IMAGE_LOAD_ERROR;
		return ret;
	} catch (std::exception& e) {
		ret->err_str = strdup(e.what());
		ret->err_code = UNKNOWN_ERROR;
		return ret;
	}

    return ret;
}

tracker_ret* get_track_position(tracker* tr) {
    Tracker* cls = (Tracker*)(tr->cls);   
    return cls->Position();
}

facesret* facerec_detect_from_file(facerec* rec, const char* file,int type) {
	facesret* ret = (facesret*)calloc(1, sizeof(facesret));
    FaceRec* cls = (FaceRec*)(rec->cls);
	image_t img;

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
    
    return cls->detect(ret, img, type);
}

facesret* facerec_detect_from_buffer(facerec* rec, unsigned char* img_data, int len,int type) {
    FaceRec* cls = (FaceRec*)(rec->cls);
	facesret* ret = (facesret*)calloc(1, sizeof(facesret));
	image_t img;

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
    
    return cls->detect(ret, img, type);
}

faceret* facerec_recognize(facerec* rec, image_pointer *p) {
    faceret* ret = (faceret*)calloc(1, sizeof(faceret));
    FaceRec* cls = (FaceRec*)(rec->cls);

    try {
        descriptor descr;
	    full_object_detection shape;
        std::tie(descr, shape) = cls->recognize(p);
        ret->descriptor = (float*)malloc(DESCR_LEN * sizeof(float));
        memcpy((uint8_t*)(ret->descriptor), (void*)&descr(0,0), DESCR_LEN * sizeof(float));

        ret->num_shape = shape.num_parts();
        ret->shape = (long*)malloc(ret->num_shape * SHAPE_LEN * sizeof(long));

        long* dst = ret->shape;
	    for (int j = 0; j < ret->num_shape; j++) {
		    dst[j*SHAPE_LEN] = shape.part(j).x() / p->upped;
		    dst[j*SHAPE_LEN+1] = shape.part(j).y() / p->upped;
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

int facerec_get_gender(facerec* rec, image_pointer *p) {
	FaceRec* cls = (FaceRec*)(rec->cls);    
	return cls->gender(p);
}

int facerec_get_age(facerec* rec, image_pointer *p) {
	FaceRec* cls = (FaceRec*)(rec->cls);    
	return cls->age(p);
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

void tracker_free(tracker* tr) {
	if (tr) {
		if (tr->cls) {
			Tracker* cls = (Tracker*)(tr->cls);
			delete cls;
			tr->cls = NULL;
		}
		free(tr);
	}
}

void image_pointer_free(image_pointer* p) {
	if (p) {
		if (p->img) {
			image_t *img = ((image_t*)p->img);
			delete img;
			p->img = NULL;
		}
	}
}