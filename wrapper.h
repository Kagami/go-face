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

typedef struct image_pointer {
    void *p;
} image_pointer;

typedef struct facesret {
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

facerec* facerec_init();
facesret* facerec_detect_file(facerec*,  image_pointer *, const char*,int);
facesret* facerec_detect_buffer(facerec*,  image_pointer *, unsigned char*, int, int);
facesret* facerec_detect_mat(facerec* rec,  image_pointer *p, const void *mat,int type);
faceret* facerec_recognize(facerec*, image_pointer*, int, int, int, int);
int facerec_gender(facerec* rec, image_pointer *p, int x, int y, int x1, int y1);
void facerec_set_cnn(facerec* , const char *);
void facerec_set_shape(facerec* , const char *);
void facerec_set_descriptor(facerec* , const char *);
void facerec_set_gender(facerec* , const char *);



void facerec_set_samples(facerec*, const float*, const int32_t*, int);
int facerec_classify(facerec*, const float*, float);
void facerec_free(facerec*);
void facerec_config_size(facerec* rec, unsigned long size);
void facerec_config_padding(facerec* rec, double padding);
void facerec_config_jittering(facerec* rec, int jittering);
void facerec_config_min_image_size(facerec* rec, int min_image_size);
void image_pointer_free(image_pointer* p);

#ifdef __cplusplus
}
#endif

#endif /* WRAPPER_H */
