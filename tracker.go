package face

// #cgo pkg-config: dlib-1
// #cgo CXXFLAGS: -std=c++1z -Wall -O3 -DNDEBUG -march=native
// #cgo LDFLAGS: -ljpeg
// #include <stdlib.h>
// #include <stdint.h>
// #include "wrapper.h"
// #include "tracker.h"
import "C"
import (
	"image"
	"unsafe"
)

type Tracker struct {
	ptr *C.tracker
}

func NewTracker() (tracker *Tracker, err error) {
	ptr := C.tracker_init()

	if ptr.err_str != nil {
		defer C.tracker_free(ptr)
		defer C.free(unsafe.Pointer(ptr.err_str))
		err = makeError(C.GoString(ptr.err_str), int(ptr.err_code))
		return
	}

	tracker = &Tracker{ptr}
	return
}

func (tracker *Tracker) Position() (rect image.Rectangle, err error) {
	ret := C.get_track_position(tracker.ptr)
	defer C.free(unsafe.Pointer(ret))

	if ret.err_str != nil {
		defer C.free(unsafe.Pointer(ret.err_str))
		err = makeError(C.GoString(ret.err_str), int(ret.err_code))
		return
	}

	// Copy faces data to Go structure.
	defer C.free(unsafe.Pointer(ret.rectangles))

	rDataLen := rectLen
	rDataPtr := unsafe.Pointer(ret.rectangles)
	rData := (*[1 << 30]C.long)(rDataPtr)[:rDataLen:rDataLen]

	x0 := int(rData[0])
	y0 := int(rData[1])
	x1 := int(rData[2])
	y1 := int(rData[3])
	rect = image.Rect(x0, y0, x1, y1)
	return
}
