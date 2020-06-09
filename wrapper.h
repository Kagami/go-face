#pragma once
#ifndef WRAPPER_H
#define WRAPPER_H

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


typedef struct tracker {
	void* cls;
	const char* err_str;
	err_code err_code;
} tracker;

typedef struct tracker_ret {
	long* rectangles;
	const char* err_str;
	err_code err_code;
} tracker_ret;

typedef struct image_pointer {
    void *img;
    void *shape;
    void *rect;
    int upped;
} image_pointer;

typedef struct facesret {
    image_pointer *p;
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

tracker* tracker_init();
facesret* start_track_from_file(tracker* tr, const char* file,int x1, int y1, int x2, int y2);
facesret* start_track_from_buffer(tracker* tr, unsigned char* img_data, int len,int x1, int y1, int x2, int y2);
facesret* update_track_from_file(tracker* tr, const char* file);
facesret* update_track_from_buffer(tracker* tr, unsigned char* img_data, int len);
tracker_ret* get_track_position(tracker* tr);

facerec* facerec_init();
facesret* facerec_detect_from_file(facerec*, const char*,int);
facesret* facerec_detect_from_buffer(facerec*, unsigned char*, int, int);
faceret* facerec_recognize(facerec*, image_pointer*);

int facerec_get_gender(facerec* rec, image_pointer *);
int facerec_get_age(facerec* rec, image_pointer *);

void facerec_set_cnn(facerec* , const char *);
void facerec_set_custom(facerec* , const char *);
void facerec_set_shape(facerec* , const char *);
void facerec_set_descriptor(facerec* , const char *);
void facerec_set_gender(facerec* , const char *);
void facerec_set_age(facerec* , const char *);

void facerec_set_samples(facerec*, const float*, const int32_t*, int);
int facerec_classify(facerec*, const float*, float);
void facerec_free(facerec*);
void tracker_free(tracker*);

void facerec_config_set_size(facerec* rec, unsigned long size);
void facerec_config_set_padding(facerec* rec, double padding);
void facerec_config_set_jittering(facerec* rec, int jittering);
void facerec_config_set_min_image_size(facerec* rec, int min_image_size);

void image_pointer_free(image_pointer* p);

#ifdef __cplusplus
}
#endif

#endif /* WRAPPER_H */
