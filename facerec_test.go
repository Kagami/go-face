package dlib

import (
	"testing"
)

func TestNumFaces(t *testing.T) {
	rec, err := NewFaceRec("testdata")
	if err != nil {
		t.Fatalf("Can't init face recognizer: %v", err)
	}
	defer rec.Close()
	ds, err := rec.GetDescriptors("testdata/pristin.jpg")
	if err != nil {
		t.Fatalf("Can't get descriptors: %v", err)
	}
	numFaces := len(ds)
	if err != nil || numFaces != 10 {
		t.Fatalf("Wrong number of faces: %d", numFaces)
	}
}
