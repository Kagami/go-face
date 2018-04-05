package dlib

// #include <stdlib.h>
// #include <stdint.h>
// #include "facerec.h"
import "C"
import (
	"image"
	"io/ioutil"
	"os"
	"unsafe"
)

const (
	rectLen  = 4
	descrLen = 128
)

// Preinitialized extractor.
type FaceRec struct {
	p *_Ctype_struct_facerec
}

// Face structure.
type Face struct {
	Rectangle  image.Rectangle
	Descriptor [descrLen]float32
}

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

func (rec *FaceRec) recognize(imgData []byte, maxFaces int) (faces []Face, err error) {
	if len(imgData) == 0 {
		err = ImageLoadError("Empty image")
		return
	}
	cImgData := (*C.uchar)(&imgData[0])
	cLen := C.int(len(imgData))
	cMaxFaces := C.int(maxFaces)
	ret := C.facerec_recognize(rec.p, cImgData, cLen, cMaxFaces)
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

	rDataLen := numFaces * rectLen
	rDataPtr := unsafe.Pointer(ret.rectangles)
	rData := (*[1 << 30]C.long)(rDataPtr)[:rDataLen:rDataLen]

	dDataLen := numFaces * descrLen
	dDataPtr := unsafe.Pointer(ret.descriptors)
	dData := (*[1 << 30]float32)(dDataPtr)[:dDataLen:dDataLen]

	for i := 0; i < numFaces; i++ {
		face := Face{}
		x0 := int(rData[i*rectLen])
		y0 := int(rData[i*rectLen+1])
		x1 := int(rData[i*rectLen+2])
		y1 := int(rData[i*rectLen+3])
		face.Rectangle = image.Rect(x0, y0, x1, y1)
		copy(face.Descriptor[:], dData[i*descrLen:(i+1)*descrLen])
		faces = append(faces, face)
	}
	return
}

// Convenient method.
func (rec *FaceRec) recognizeFile(imgPath string, maxFaces int) (face []Face, err error) {
	fd, err := os.Open(imgPath)
	if err != nil {
		return
	}
	imgData, err := ioutil.ReadAll(fd)
	if err != nil {
		return
	}
	return rec.recognize(imgData, maxFaces)
}

// Recognize all image faces.
// Empty list is returned if there are no faces, error is returned if
// there was some error while decoding/processing image.
func (rec *FaceRec) Recognize(imgData []byte) (faces []Face, err error) {
	return rec.recognize(imgData, 0)
}

// Recognize if image has single face or return nil otherwise.
func (rec *FaceRec) RecognizeSingle(imgData []byte) (face *Face, err error) {
	faces, err := rec.recognize(imgData, 1)
	if err != nil || len(faces) != 1 {
		return
	}
	face = &faces[0]
	return
}

func (rec *FaceRec) RecognizeFile(imgPath string) (faces []Face, err error) {
	return rec.recognizeFile(imgPath, 0)
}

func (rec *FaceRec) RecognizeSingleFile(imgPath string) (face *Face, err error) {
	faces, err := rec.recognizeFile(imgPath, 1)
	if err != nil || len(faces) != 1 {
		return
	}
	face = &faces[0]
	return
}

func (rec *FaceRec) Close() {
	C.facerec_free(rec.p)
	rec.p = nil
}
