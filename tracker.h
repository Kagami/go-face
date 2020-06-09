#pragma once
#ifndef TRACKER_H
#define TRACKER_H

#ifdef __cplusplus
#include <shared_mutex>
#include <dlib/dnn.h>
#include <dlib/image_loader/image_loader.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/graph_utils.h>
#include "facerec.h"

typedef dlib::correlation_tracker correlation_tracker;

class Tracker {   
private:  
	correlation_tracker tracker;   
public:
	Tracker();
    void StartTrack(image_t &,rectangle);
    void Update(image_t &img);
    tracker_ret *Position();
};

#endif

#endif /* TRACKER_H */
