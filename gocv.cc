// +build gocv

#include <dlib/opencv.h> 
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include "facerec.h"
#include "gocv.h"

using namespace dlib;

facesret* facerec_detect_from_mat(facerec* rec, image_pointer *p, const void *mat,int type) {
	facesret* ret = (facesret*)calloc(1, sizeof(facesret));
	image_t img;
	std::vector<rectangle> rects;
    cv::Mat *mat_img = new cv::Mat(*((cv::Mat*)mat));
    IplImage ipl_img = cvIplImage(*mat_img);
    
	try {
		cv_image<bgr_pixel> dlib_img(&ipl_img);
        assign_image(img, dlib_img);
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
