package face

// #cgo pkg-config: dlib-1
// #cgo CXXFLAGS: -std=c++1z -Wall -O3 -DNDEBUG -march=native
// #cgo LDFLAGS: -ljpeg
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
	shapeLen = 2
	rectLen  = 4
	descrLen = 128
)

// A Recognizer creates face descriptors for provided images and
// classifies them into categories.
type Recognizer struct {
	ptr *C.facerec
}

// Face holds coordinates and descriptor of the human face.
type Face struct {
	Rectangle  image.Rectangle
	Shapes     []image.Point
	Descriptor Descriptor
}

// Descriptor holds 128-dimensional feature vector.
type Descriptor [128]float32

// New creates new face with the provided parameters.
func New(r image.Rectangle, s []image.Point, d Descriptor) Face {
	return Face{r, s, d}
}

// NewRecognizer returns a new recognizer interface. modelDir points to
// directory with shape_predictor_5_face_landmarks.dat and
// dlib_face_recognition_resnet_model_v1.dat files.
func NewRecognizer(modelDir string) (rec *Recognizer, err error) {
	cModelDir := C.CString(modelDir)
	defer C.free(unsafe.Pointer(cModelDir))
	ptr := C.facerec_init(cModelDir)

	if ptr.err_str != nil {
		defer C.facerec_free(ptr)
		defer C.free(unsafe.Pointer(ptr.err_str))
		err = makeError(C.GoString(ptr.err_str), int(ptr.err_code))
		return
	}

	rec = &Recognizer{ptr}
	return
}

func (rec *Recognizer) Config(size int, padding float32, jittering int) {
	cSize := C.ulong(size)
	cPadding := C.double(padding)
	cJittering := C.int(jittering)
	C.facerec_config(rec.ptr, cSize, cPadding, cJittering)
}

func (rec *Recognizer) recognize(imgData []byte, maxFaces int) (faces []Face, err error) {
	if len(imgData) == 0 {
		err = ImageLoadError("Empty image")
		return
	}
	cImgData := (*C.uint8_t)(&imgData[0])
	cLen := C.int(len(imgData))
	cMaxFaces := C.int(maxFaces)

	ret := C.facerec_recognize(rec.ptr, cImgData, cLen, cMaxFaces)
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

func Squared_euclidean_distance(sample Descriptor, testSample Descriptor) (distance float32) {
	cTestSample := (*C.float)(unsafe.Pointer(&testSample))
	cSample := (*C.float)(unsafe.Pointer(&sample))

	return float32(C.squared_euclidean_distance(cSample, cTestSample))
}

func (rec *Recognizer) recognizeFile(imgPath string, maxFaces int) (face []Face, err error) {
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

// Recognize returns all faces found on the provided image, sorted from
// left to right. Empty list is returned if there are no faces, error is
// returned if there was some error while decoding/processing image.
// Only JPEG format is currently supported. Thread-safe.
func (rec *Recognizer) Recognize(imgData []byte) (faces []Face, err error) {
	return rec.recognize(imgData, 0)
}

// RecognizeSingle returns face if it's the only face on the image or
// nil otherwise. Only JPEG format is currently supported. Thread-safe.
func (rec *Recognizer) RecognizeSingle(imgData []byte) (face *Face, err error) {
	faces, err := rec.recognize(imgData, 1)
	if err != nil || len(faces) != 1 {
		return
	}
	face = &faces[0]
	return
}

// Same as Recognize but accepts image path instead.
func (rec *Recognizer) RecognizeFile(imgPath string) (faces []Face, err error) {
	return rec.recognizeFile(imgPath, 0)
}

// Same as RecognizeSingle but accepts image path instead.
func (rec *Recognizer) RecognizeSingleFile(imgPath string) (face *Face, err error) {
	faces, err := rec.recognizeFile(imgPath, 1)
	if err != nil || len(faces) != 1 {
		return
	}
	face = &faces[0]
	return
}

// SetSamples sets known descriptors so you can classify the new ones.
// Thread-safe.
func (rec *Recognizer) SetSamples(samples []Descriptor, cats []int32) {
	if len(samples) == 0 || len(samples) != len(cats) {
		return
	}
	cSamples := (*C.float)(unsafe.Pointer(&samples[0]))
	cCats := (*C.int32_t)(unsafe.Pointer(&cats[0]))
	cLen := C.int(len(samples))
	C.facerec_set_samples(rec.ptr, cSamples, cCats, cLen)
}

// Classify returns class ID for the given descriptor. Negative index is
// returned if no match. Thread-safe.
func (rec *Recognizer) Classify(testSample Descriptor) int {
	cTestSample := (*C.float)(unsafe.Pointer(&testSample))
	return int(C.facerec_classify(rec.ptr, cTestSample, -1))
}

// Same as Classify but allows to specify max distance between faces to
// consider it a match. Start with 0.6 if not sure.
func (rec *Recognizer) ClassifyThreshold(testSample Descriptor, tolerance float32) int {
	cTestSample := (*C.float)(unsafe.Pointer(&testSample))
	cTolerance := C.float(tolerance)
	return int(C.facerec_classify(rec.ptr, cTestSample, cTolerance))
}

// Close frees resources taken by the Recognizer. Safe to call multiple
// times. Don't use Recognizer after close call.
func (rec *Recognizer) Close() {
	C.facerec_free(rec.ptr)
	rec.ptr = nil
}
