package face

import (
	"fmt"
	"log"
)

const (
	// Path to directory with models.
	modelDir = "testdata"

	// Test image with 10 faces.
	testImageTenFaces = "testdata/pristin.jpg"
)

// This example shows the basic usage of the package: create an
// recognizer, recognize faces, classify them using few known ones.
func Example() {
	rec, err := NewRecognizer(modelDir)
	if err != nil {
		log.Fatalf("Can't init face recognizer: %v", err)
	}

	faces, err := rec.RecognizeFile(testImageTenFaces)
	if err != nil {
		log.Fatalf("Can't get faces: %v", err)
	}
	numFaces := len(faces)
	if numFaces != 10 {
		log.Fatalf("Wrong number of faces: %d", numFaces)
	}
	fmt.Printf("Faces on %s: %v", testImageTenFaces, faces)
}
