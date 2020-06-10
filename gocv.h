// +build gocv

#pragma once
#ifndef GOCV_H
#define GOCV_H

#include "wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

facesret* facerec_detect_from_mat(facerec* rec, const void *mat,int type);

facesret* start_track_from_mat(tracker* tr, const void *mat,int x1, int y1, int x2, int y2);
update_ret* update_track_from_mat(tracker* tr, const void *mat);

#ifdef __cplusplus
}
#endif

#endif /* GOCV_H */
