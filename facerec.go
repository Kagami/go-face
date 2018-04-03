package dlib

// #include <stdlib.h>
// #include "facerec.h"
import "C"
import (
	"unsafe"
)

const (
	DESCR_LEN = 128
)

// Preinitialized extractor.
type FaceRec struct {
	p *_Ctype_struct_facerec
}

// Descriptor alias.
type Descriptor [DESCR_LEN]float32

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

func (rec *FaceRec) getDescriptors(imgPath string, maxFaces int) (ds []Descriptor, err error) {
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

	// Copy descriptor data to Go structure.
	defer C.free(unsafe.Pointer(ret.descriptors))
	dataLen := numFaces * DESCR_LEN
	dataPtr := unsafe.Pointer(ret.descriptors)
	data := (*[1 << 30]float32)(dataPtr)[:dataLen:dataLen]
	for i := 0; i < numFaces; i++ {
		var d Descriptor
		copy(d[:], data[i*DESCR_LEN:(i+1)*DESCR_LEN])
		ds = append(ds, d)
	}
	return
}

// Get face descriptor if image has single face or nil otherwise.
func (rec *FaceRec) GetDescriptor(imgPath string) (d *Descriptor, err error) {
	ds, err := rec.getDescriptors(imgPath, 1)
	if err != nil {
		return
	}
	if len(ds) != 1 {
		return
	}
	d = &ds[0]
	return
}

// Get face descriptors from the provided image file.
// Empty list is returned if there are no faces, error is returned if
// there was some error while decoding/processing image.
func (rec *FaceRec) GetDescriptors(imgPath string) (ds []Descriptor, err error) {
	return rec.getDescriptors(imgPath, 0)
}

func (rec *FaceRec) Close() {
	C.facerec_free(rec.p)
	rec.p = nil
}
