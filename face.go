package face

// #cgo !windows pkg-config: opencv4
// #cgo CXXFLAGS:   --std=c++11
// #cgo windows  CPPFLAGS:   -IC:/opencv/build/install/include
// #cgo windows  LDFLAGS:    -LC:/opencv/build/install/x64/mingw/lib -lopencv_core420 -lopencv_face420 -lopencv_videoio420 -lopencv_imgproc420 -lopencv_highgui420 -lopencv_imgcodecs420 -lopencv_objdetect420 -lopencv_features2d420 -lopencv_video420 -lopencv_dnn420 -lopencv_xfeatures2d420 -lopencv_plot420 -lopencv_tracking420 -lopencv_img_hash420 -lopencv_calib3d420
// #cgo pkg-config: dlib-1
// #cgo CXXFLAGS: -std=c++1z -Wall -O3 -DNDEBUG -march=native
// #cgo LDFLAGS: -ljpeg
// #cgo LDFLAGS: -lopenblas
// #cgo LDFLAGS: -lquadmath
// #cgo LDFLAGS: -lgfortran
// #include <stdlib.h>
// #include <stdint.h>
// #include "facerec.h"
import "C"
import (
	"errors"
	"fmt"
	"image"
	"math"
	"os"
	"unsafe"

	"gocv.io/x/gocv"
)

const (
	rectLen  = 4
	descrLen = 128
	shapeLen = 2
)

// A Recognizer creates face descriptors for provided images and
// classifies them into categories.
type Recognizer struct {
	ptr *C.facerec
}

// Face holds coordinates and descriptor of the human face.
type Face struct {
	imagePointer *C.image_pointer
	Rectangle    image.Rectangle
	Descriptor   Descriptor
	Shapes       []image.Point
}

// Descriptor holds 128-dimensional feature vector.
type Descriptor [128]float32

func SquaredEuclideanDistance(d1 Descriptor, d2 Descriptor) (sum float64) {
	for i := range d1 {
		sum = sum + math.Pow(float64(d2[i]-d1[i]), 2)
	}

	return sum
}

// New creates new face with the provided parameters.
func New(r image.Rectangle, d Descriptor) Face {
	return Face{Rectangle: r, Descriptor: d, Shapes: []image.Point{}}
}

func NewWithShape(r image.Rectangle, s []image.Point, d Descriptor) Face {
	return Face{Rectangle: r, Descriptor: d, Shapes: s}
}

// NewRecognizer returns a new recognizer interface. modelDir points to
// directory with shape_predictor_5_face_landmarks.dat and
// dlib_face_recognition_resnet_model_v1.dat files.
func NewRecognizer(resnetPath, cnnResnetPath, shapePredictorPath string) (rec *Recognizer, err error) {
	if !fileExists(resnetPath) {
		err = errors.New(fmt.Sprintf("File '%s' not found!", resnetPath))
		return
	}
	if !fileExists(cnnResnetPath) {
		err = errors.New(fmt.Sprintf("File '%s' not found!", cnnResnetPath))
		return
	}
	if !fileExists(shapePredictorPath) {
		err = errors.New(fmt.Sprintf("File '%s' not found!", shapePredictorPath))
		return
	}
	cResnetPath := C.CString(resnetPath)
	defer C.free(unsafe.Pointer(cResnetPath))
	cCnnResnetPath := C.CString(cnnResnetPath)
	defer C.free(unsafe.Pointer(cCnnResnetPath))
	cShapePredictorPath := C.CString(shapePredictorPath)
	defer C.free(unsafe.Pointer(cShapePredictorPath))

	ptr := C.facerec_init(cResnetPath, cCnnResnetPath, cShapePredictorPath)

	if ptr.err_str != nil {
		defer C.facerec_free(ptr)
		defer C.free(unsafe.Pointer(ptr.err_str))
		err = makeError(C.GoString(ptr.err_str), int(ptr.err_code))
		return
	}

	rec = &Recognizer{ptr}
	return
}

func NewRecognizerWithConfig(resnetPath, cnnResnetPath, shapePredictorPath string, size int, padding float32, jittering int, minImageSize int) (rec *Recognizer, err error) {
	rec, err = NewRecognizer(resnetPath, cnnResnetPath, shapePredictorPath)
	cSize := C.ulong(size)
	cPadding := C.double(padding)
	cJittering := C.int(jittering)
	cMinImageSize := C.int(minImageSize)
	C.facerec_config(rec.ptr, cSize, cPadding, cJittering, cMinImageSize)
	return
}

func (rec *Recognizer) detectBuffer(type_ int, imgData []byte) (faces []Face, err error) {
	if len(imgData) == 0 {
		err = ImageLoadError("Empty image")
		return
	}
	cImgData := (*C.uchar)(&imgData[0])
	cLen := C.int(len(imgData))
	cType := C.int(type_)
	var ptr C.image_pointer

	ret := C.facerec_detect_buffer(rec.ptr, (*C.image_pointer)(unsafe.Pointer(&ptr)), cImgData, cLen, cType)
	defer C.free(unsafe.Pointer(ret))

	if ret.err_str != nil {
		defer C.free(unsafe.Pointer(ret.err_str))
		err = makeError(C.GoString(ret.err_str), int(ret.err_code))
		return
	}

	numFaces := int(ret.num_faces)
	if numFaces == 0 {
		return
	}

	// Copy faces data to Go structure.
	defer C.free(unsafe.Pointer(ret.rectangles))

	rDataLen := numFaces * rectLen
	rDataPtr := unsafe.Pointer(ret.rectangles)
	rData := (*[1 << 30]C.long)(rDataPtr)[:rDataLen:rDataLen]

	for i := 0; i < numFaces; i++ {
		fmt.Println("pointer", &ptr)
		face := Face{imagePointer: &ptr}
		x0 := int(rData[i*rectLen])
		y0 := int(rData[i*rectLen+1])
		x1 := int(rData[i*rectLen+2])
		y1 := int(rData[i*rectLen+3])
		face.Rectangle = image.Rect(x0, y0, x1, y1)
		faces = append(faces, face)
	}
	return
}

func (rec *Recognizer) detectMat(type_ int, mat gocv.Mat) (faces []Face, err error) {
	cType := C.int(type_)
	var ptr C.image_pointer

	ret := C.facerec_detect_mat(rec.ptr, (*C.image_pointer)(unsafe.Pointer(&ptr)), unsafe.Pointer(mat.Ptr()), cType)
	defer C.free(unsafe.Pointer(ret))

	if ret.err_str != nil {
		defer C.free(unsafe.Pointer(ret.err_str))
		err = makeError(C.GoString(ret.err_str), int(ret.err_code))
		return
	}

	numFaces := int(ret.num_faces)
	if numFaces == 0 {
		return
	}

	// Copy faces data to Go structure.
	defer C.free(unsafe.Pointer(ret.rectangles))

	rDataLen := numFaces * rectLen
	rDataPtr := unsafe.Pointer(ret.rectangles)
	rData := (*[1 << 30]C.long)(rDataPtr)[:rDataLen:rDataLen]

	for i := 0; i < numFaces; i++ {
		face := Face{imagePointer: &ptr}
		x0 := int(rData[i*rectLen])
		y0 := int(rData[i*rectLen+1])
		x1 := int(rData[i*rectLen+2])
		y1 := int(rData[i*rectLen+3])
		face.Rectangle = image.Rect(x0, y0, x1, y1)
		faces = append(faces, face)
	}
	return
}

func (rec *Recognizer) detectFile(type_ int, file string) (faces []Face, err error) {
	if !fileExists(file) {
		err = ImageLoadError(fmt.Sprintf("File '%s' not found!", file))
		return
	}

	cType := C.int(type_)
	cFile := C.CString(file)
	defer C.free(unsafe.Pointer(cFile))
	var ptr *C.image_pointer
	ret := C.facerec_detect_file(rec.ptr, ptr, cFile, cType)
	defer C.free(unsafe.Pointer(ret))

	if ret.err_str != nil {
		defer C.free(unsafe.Pointer(ret.err_str))
		err = makeError(C.GoString(ret.err_str), int(ret.err_code))
		return
	}

	numFaces := int(ret.num_faces)
	if numFaces == 0 {
		return
	}

	// Copy faces data to Go structure.
	defer C.free(unsafe.Pointer(ret.rectangles))

	rDataLen := numFaces * rectLen
	rDataPtr := unsafe.Pointer(ret.rectangles)
	rData := (*[1 << 30]C.long)(rDataPtr)[:rDataLen:rDataLen]

	for i := 0; i < numFaces; i++ {
		face := Face{imagePointer: ptr}
		x0 := int(rData[i*rectLen])
		y0 := int(rData[i*rectLen+1])
		x1 := int(rData[i*rectLen+2])
		y1 := int(rData[i*rectLen+3])
		face.Rectangle = image.Rect(x0, y0, x1, y1)
		faces = append(faces, face)
	}
	return
}

func (rec *Recognizer) DetectMat(mat gocv.Mat) (faces []Face, err error) {
	return rec.detectMat(0, mat)
}

func (rec *Recognizer) DetectMatCNN(mat gocv.Mat) (faces []Face, err error) {
	return rec.detectMat(1, mat)
}

// Detect returns all faces found on the provided image, sorted from
// left to right. Empty list is returned if there are no faces, error is
// returned if there was some error while decoding/processing image.
// Only JPEG format is currently supported. Thread-safe.
func (rec *Recognizer) Detect(imgData []byte) (faces []Face, err error) {
	return rec.detectBuffer(0, imgData)
}

func (rec *Recognizer) DetectCNN(imgData []byte) (faces []Face, err error) {
	return rec.detectBuffer(1, imgData)
}

// Same as Recognize but accepts image path instead.
func (rec *Recognizer) DetectFromFile(imgPath string) (faces []Face, err error) {
	return rec.detectFile(0, imgPath)
}

func (rec *Recognizer) DetectFromFileCNN(imgPath string) (faces []Face, err error) {
	return rec.detectFile(1, imgPath)
}

func (rec *Recognizer) Recognize(face *Face) error {
	x := C.int(face.Rectangle.Min.X)
	y := C.int(face.Rectangle.Min.Y)
	x1 := C.int(face.Rectangle.Max.X)
	y1 := C.int(face.Rectangle.Max.Y)

	ret := C.facerec_recognize(rec.ptr, (*C.image_pointer)(unsafe.Pointer(face.imagePointer)), x, y, x1, y1)
	defer C.free(unsafe.Pointer(ret))

	if ret.err_str != nil {
		defer C.free(unsafe.Pointer(ret.err_str))
		err := makeError(C.GoString(ret.err_str), int(ret.err_code))
		return err
	}
	numShapes := int(ret.num_shape)
	defer C.free(unsafe.Pointer(ret.shape))
	defer C.free(unsafe.Pointer(ret.descriptor))

	dDataPtr := unsafe.Pointer(ret.descriptor)
	dData := (*[1 << 30]float32)(dDataPtr)[:descrLen:descrLen]

	sDataLen := numShapes * shapeLen
	sDataPtr := unsafe.Pointer(ret.shape)
	sData := (*[1 << 30]C.long)(sDataPtr)[:sDataLen:sDataLen]

	copy(face.Descriptor[:], dData[:descrLen])
	for j := 0; j < numShapes; j++ {
		shapeX := int(sData[(j)*shapeLen])
		shapeY := int(sData[(j)*shapeLen+1])
		face.Shapes = append(face.Shapes, image.Point{shapeX, shapeY})
	}

	return nil
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

func fileExists(filename string) bool {
	info, err := os.Stat(filename)
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir()
}

func (f *Face) Close() {
	f.imagePointer = nil
}
