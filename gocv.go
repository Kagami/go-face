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
	"image/color"
	"unsafe"

	"gocv.io/x/gocv"
)

func (rec *Recognizer) detectFromMat(type_ int, mat gocv.Mat) (faces []Face, err error) {
	cType := C.int(type_)

	ret := C.facerec_detect_from_mat(rec.ptr, unsafe.Pointer(mat.Ptr()), cType)
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
	defer C.free(unsafe.Pointer(ret.p))

	rPtr := unsafe.Pointer(ret.p)
	rP := (*[1 << 30]C.image_pointer)(rPtr)[:numFaces:numFaces]
	rDataLen := numFaces * rectLen
	rDataPtr := unsafe.Pointer(ret.rectangles)
	rData := (*[1 << 30]C.long)(rDataPtr)[:rDataLen:rDataLen]

	for i := 0; i < numFaces; i++ {
		face := Face{imagePointer: rP[i]}
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

func RenderFaceDetections(img *gocv.Mat, Shapes []image.Point, col color.RGBA, thickness int) {
	if len(Shapes) == 5 {
		gocv.Line(img, Shapes[0], Shapes[1], col, thickness)
		gocv.Line(img, Shapes[1], Shapes[4], col, thickness)
		gocv.Line(img, Shapes[4], Shapes[3], col, thickness)
		gocv.Line(img, Shapes[3], Shapes[2], col, thickness)
	} else {
		// Around Chin. Ear to Ear
		for i := 1; i <= 16; i++ {
			gocv.Line(img, Shapes[i], Shapes[i-1], col, thickness)
		}

		// Line on top of nose
		for i := 28; i <= 30; i++ {
			gocv.Line(img, Shapes[i], Shapes[i-1], col, thickness)
		}

		// left eyebrow
		for i := 18; i <= 21; i++ {
			gocv.Line(img, Shapes[i], Shapes[i-1], col, thickness)
		}
		// Right eyebrow
		for i := 23; i <= 26; i++ {
			gocv.Line(img, Shapes[i], Shapes[i-1], col, thickness)
		}
		// Bottom part of the nose
		for i := 31; i <= 35; i++ {
			gocv.Line(img, Shapes[i], Shapes[i-1], col, thickness)
		}
		// Line from the nose to the bottom part above
		gocv.Line(img, Shapes[30], Shapes[35], col, thickness)

		// Left eye
		for i := 37; i <= 41; i++ {
			gocv.Line(img, Shapes[i], Shapes[i-1], col, thickness)
		}
		gocv.Line(img, Shapes[36], Shapes[41], col, thickness)

		// Right eye
		for i := 43; i <= 47; i++ {
			gocv.Line(img, Shapes[i], Shapes[i-1], col, thickness)
		}
		gocv.Line(img, Shapes[42], Shapes[47], col, thickness)

		// Lips outer part
		for i := 49; i <= 59; i++ {
			gocv.Line(img, Shapes[i], Shapes[i-1], col, thickness)
		}
		gocv.Line(img, Shapes[48], Shapes[59], col, thickness)

		// Lips inside part
		for i := 61; i <= 67; i++ {
			gocv.Line(img, Shapes[i], Shapes[i-1], col, thickness)
		}
		gocv.Line(img, Shapes[60], Shapes[67], col, thickness)
	}
}
