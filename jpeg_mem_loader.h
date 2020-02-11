#pragma once
#ifndef JPEG_MEM_LOADER_H
#define JPEG_MEM_LOADER_H

void load_mem_jpeg(dlib::matrix<dlib::rgb_pixel>& img, const uint8_t* img_data, int len);

#endif /* JPEG_MEM_LOADER_H */
