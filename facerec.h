#pragma once

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
} facerec;

typedef struct faceret {
	int num_faces;
	long* rectangles;
	float* descriptors;
	const char* err_str;
	err_code err_code;
} faceret;

facerec* facerec_init(const char* model_dir);
faceret* facerec_recognize(facerec* rec, const uint8_t* img_data, int len, int max_faces);
int facerec_classify(facerec* rec, const float* descriptors, int len, const float* descriptor);
void facerec_free(facerec* rec);

#ifdef __cplusplus
}
#endif
