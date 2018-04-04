package dlib

// #include <stdint.h>
// #include "facerec.h"
import "C"

type ImageLoadError string

func (e ImageLoadError) Error() string {
	return string(e)
}

type SerializationError string

func (e SerializationError) Error() string {
	return string(e)
}

type RecognizeError string

func (e RecognizeError) Error() string {
	return string(e)
}

type UnknownError string

func (e UnknownError) Error() string {
	return string(e)
}

// Construct Go error from passed error info.
func makeError(s string, code int) error {
	switch code {
	case C.IMAGE_LOAD_ERROR:
		return ImageLoadError(s)
	case C.SERIALIZATION_ERROR:
		return SerializationError(s)
	case C.RECOGNIZE_ERROR:
		return RecognizeError(s)
	default:
		return UnknownError(s)
	}
}
