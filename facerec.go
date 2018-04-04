package dlib

// #include <stdlib.h>
// #include "facerec.h"
import "C"
import (
	"unsafe"
)

const (
	descrLen = 128
)

// Preinitialized extractor.
type FaceRec struct {
	p *_Ctype_struct_facerec
}

// Face structure.
type Face struct {
	Rectangle  [4]int32
	Descriptor FaceDescriptor
}

// Descriptor alias.
type FaceDescriptor [descrLen]float32

func NewFaceRec(modelDir string) (rec *FaceRec, err error) {
	cModelDir := C.CString(modelDir)
	defer C.free(unsafe.Pointer(cModelDir))
	p := C.facerec_init(cModelDir)

	if p.err_str != nil {
		defer C.facerec_free(p)
		defer C.free(unsafe.Pointer(p.err_str))
		err = makeError(C.GoString(p.err_str), int(p.err_code))
		return
	}

	rec = &FaceRec{p}
	return
}

func (rec *FaceRec) recognize(imgPath string, maxFaces int) (faces []Face, err error) {
	cImgPath := C.CString(imgPath)
	defer C.free(unsafe.Pointer(cImgPath))
	ret := C.facerec_recognize(rec.p, cImgPath, C.int(maxFaces))
	defer C.free(unsafe.Pointer(ret))

	if ret.err_str != nil {
		defer C.free(unsafe.Pointer(ret.err_str))
		err = makeError(C.GoString(ret.err_str), int(ret.err_code))
		return
	}

	// No faces.
	numFaces := int(ret.num_faces)
	if numFaces == 0 {
		return
	}

	// Copy faces data to Go structure.
	defer C.free(unsafe.Pointer(ret.rectangles))
	defer C.free(unsafe.Pointer(ret.descriptors))

	rDataLen := numFaces * 4
	rDataPtr := unsafe.Pointer(ret.rectangles)
	rData := (*[1 << 30]C.long)(rDataPtr)[:rDataLen:rDataLen]

	dDataLen := numFaces * descrLen
	dDataPtr := unsafe.Pointer(ret.descriptors)
	dData := (*[1 << 30]float32)(dDataPtr)[:dDataLen:dDataLen]

	for i := 0; i < numFaces; i++ {
		face := Face{}
		face.Rectangle[0] = int32(rData[i*4])
		face.Rectangle[1] = int32(rData[i*4+1])
		face.Rectangle[2] = int32(rData[i*4+2])
		face.Rectangle[3] = int32(rData[i*4+3])
		copy(face.Descriptor[:], dData[i*descrLen:(i+1)*descrLen])
		faces = append(faces, face)
	}
	return
}

// Recognize all image faces.
// Empty list is returned if there are no faces, error is returned if
// there was some error while decoding/processing image.
func (rec *FaceRec) Recognize(imgPath string) (faces []Face, err error) {
	return rec.recognize(imgPath, 0)
}

// Recognize if image has single face or return nil otherwise.
func (rec *FaceRec) RecognizeSingle(imgPath string) (face *Face, err error) {
	faces, err := rec.recognize(imgPath, 1)
	if err != nil {
		return
	}
	if len(faces) != 1 {
		return
	}
	face = &faces[0]
	return
}

func (rec *FaceRec) Close() {
	C.facerec_free(rec.p)
	rec.p = nil
}
