#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	IMAGE_LOAD_ERROR,
	SERIALIZATION_ERROR,
	RECOGNIZE_ERROR,
	UNKNOWN_ERROR,
} err_code;

typedef struct facerec {
	void* cls;
	const char* err_str;
	err_code err_code;
} facerec;

typedef struct faceret {
	int num_faces;
	int32_t* rectangles;
	float* descriptors;
	const char* err_str;
	err_code err_code;
} faceret;

facerec* facerec_init(const char* model_dir);
faceret* facerec_recognize(facerec* rec, const char* img_path, int max_faces);
void facerec_free(facerec* rec);

#ifdef __cplusplus
}
#endif
