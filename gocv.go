// +build gocv

package face

// #cgo !windows pkg-config: opencv4
// #cgo CXXFLAGS:   --std=c++1z
// #cgo windows  CPPFLAGS:   -IC:/opencv/build/install/include
// #cgo windows  LDFLAGS:    -LC:/opencv/build/install/x64/mingw/lib -lopencv_core420 -lopencv_face420 -lopencv_videoio420 -lopencv_imgproc420 -lopencv_highgui420 -lopencv_imgcodecs420 -lopencv_objdetect420 -lopencv_features2d420 -lopencv_video420 -lopencv_dnn420 -lopencv_xfeatures2d420 -lopencv_plot420 -lopencv_tracking420 -lopencv_img_hash420 -lopencv_calib3d420
// #include <stdlib.h>
// #include <stdint.h>
// #include "wrapper.h"
// #include "gocv.h"
import "C"
import (
	"image"
	"unsafe"

	"gocv.io/x/gocv"
)

func (rec *Recognizer) detectFromMat(type_ int, mat gocv.Mat) (faces []Face, err error) {
	cType := C.int(type_)
	var ptr C.image_pointer

	ret := C.facerec_detect_from_mat(rec.ptr, (*C.image_pointer)(unsafe.Pointer(&ptr)), unsafe.Pointer(mat.Ptr()), cType)
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

func (rec *Recognizer) DetectFromMat(mat gocv.Mat) (faces []Face, err error) {
	return rec.detectFromMat(0, mat)
}

func (rec *Recognizer) DetectFromMatCNN(mat gocv.Mat) (faces []Face, err error) {
	return rec.detectFromMat(1, mat)
}
