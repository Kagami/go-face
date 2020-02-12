#pragma once
#ifndef TEMPLATE_H
#define TEMPLATE_H

#ifdef __cplusplus
#include <shared_mutex>
#include <dlib/dnn.h>
#include <dlib/image_loader/image_loader.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/graph_utils.h>


// don't kill me))
using namespace dlib;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

using cnn_anet_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;


template <int N, template <typename> class BN, int stride, typename SUBNET>
using block_gender = BN<con<N, 3, 3, stride, stride, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using res_ = relu<block_gender<N, bn_con, 1, SUBNET>>;
template <int N, typename SUBNET> using ares_ = relu<block_gender<N, affine, 1, SUBNET>>;

template <typename SUBNET> using alevel1_ = avg_pool<2, 2, 2, 2, ares_<64, SUBNET>>;
template <typename SUBNET> using alevel2_ = avg_pool<2, 2, 2, 2, ares_<32, SUBNET>>;

using agender_type = loss_multiclass_log<fc<2, multiply<relu<fc<16, multiply<alevel1_<alevel2_< input_rgb_image_sized<32>>>>>>>>>;

const unsigned long number_of_age_classes = 81;

// The resnet basic block.
template<
	int num_filters,
	template<typename> class BN,  // some kind of batch normalization or affine layer
	int stride,
	typename SUBNET
>
using basicblock = BN<con<num_filters, 3, 3, 1, 1, relu<BN<con<num_filters, 3, 3, stride, stride, SUBNET>>>>>;

// A residual making use of the skip layer mechanism.
template<
	template<int, template<typename> class, int, typename> class BLOCK,  // a basic block defined before
	int num_filters,
	template<typename> class BN,  // some kind of batch normalization or affine layer
	typename SUBNET
> // adds the block to the result of tag1 (the subnet)
using residual = add_prev1<BLOCK<num_filters, BN, 1, tag1<SUBNET>>>;

// A residual that does subsampling (we need to subsample the output of the subnet, too).
template<
	template<int, template<typename> class, int, typename> class BLOCK,  // a basic block defined before
	int num_filters,
	template<typename> class BN,
	typename SUBNET
>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<BLOCK<num_filters, BN, 2, tag1<SUBNET>>>>>>;

// Residual block with optional downsampling and batch normalization.
template<
	template<template<int, template<typename> class, int, typename> class, int, template<typename>class, typename> class RESIDUAL,
	template<int, template<typename> class, int, typename> class BLOCK,
	int num_filters,
	template<typename> class BN,
	typename SUBNET
>
using residual_block = relu<RESIDUAL<BLOCK, num_filters, BN, SUBNET>>;

template<int num_filters, typename SUBNET>
using aresbasicblock_down = residual_block<residual_down, basicblock, num_filters, affine, SUBNET>;

// Some useful definitions to design the affine versions for inference.
template<typename SUBNET> using aresbasicblock256 = residual_block<residual, basicblock, 256, affine, SUBNET>;
template<typename SUBNET> using aresbasicblock128 = residual_block<residual, basicblock, 128, affine, SUBNET>;
template<typename SUBNET> using aresbasicblock64  = residual_block<residual, basicblock, 64, affine, SUBNET>;

// Common input for standard resnets.
template<typename INPUT>
using aresnet_input = max_pool<3, 3, 2, 2, relu<affine<con<64, 7, 7, 2, 2, INPUT>>>>;

// Resnet-10 architecture for estimating.
template<typename SUBNET>
using aresnet10_level1 = aresbasicblock256<aresbasicblock_down<256, SUBNET>>;
template<typename SUBNET>
using aresnet10_level2 = aresbasicblock128<aresbasicblock_down<128, SUBNET>>;
template<typename SUBNET>
using aresnet10_level3 = aresbasicblock64<SUBNET>;
// The resnet 10 backbone.
template<typename INPUT>
using aresnet10_backbone = avg_pool_everything<
	aresnet10_level1<
	aresnet10_level2<
	aresnet10_level3<
	aresnet_input<INPUT>>>>>;

using apredictor_t = loss_multiclass_log<fc<number_of_age_classes, aresnet10_backbone<input_rgb_image>>>;

#endif
#endif /* TEMPLATE_H */
