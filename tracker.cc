#include <shared_mutex>
#include <dlib/dnn.h>
#include <dlib/image_loader/image_loader.h>
#include <dlib/image_loader/jpeg_loader.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/graph_utils.h>
#include <dlib/image_io.h>
#include "tracker.h"
#include "facerec.h"

using namespace dlib;

Tracker::Tracker() {
}

void Tracker::StartTrack(image_t &img, rectangle rect) {
    tracker.start_track(img, rect);
}


double Tracker::Update(image_t &img) {
    return tracker.update(img);
}

tracker_ret *Tracker::Position() {
    	tracker_ret* ret = (tracker_ret*)calloc(1, sizeof(tracker_ret));

    auto rect = tracker.get_position();
    
    	ret->rectangles = (long*)malloc(RECT_LEN * sizeof(long));
        
    long* dst = ret->rectangles;
    dst[0] = rect.left();
    dst[1] = rect.top();
    dst[2] = rect.right();
    dst[3] = rect.bottom();
        
    return ret;
}
