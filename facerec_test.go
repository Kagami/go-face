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
	faces, err := rec.Recognize("testdata/pristin.jpg")
	if err != nil {
		t.Fatalf("Can't recognize faces: %v", err)
	}
	numFaces := len(faces)
	if err != nil || numFaces != 10 {
		t.Fatalf("Wrong number of faces: %d", numFaces)
	}
}
