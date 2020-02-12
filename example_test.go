package face_test

import (
	"fmt"
	"log"
	"path/filepath"

	"github.com/Kagami/go-face"
)

// Path to directory with models and test images. Here it's assumed it
// points to the <https://github.com/Kagami/go-face-testdata> clone.
const cnn = "testdata/mmod_human_face_detector.dat"
const shape = "testdata/shape_predictor_68_face_landmarks.dat"
const descr = "testdata/dlib_face_recognition_resnet_model_v1.dat"
const gender = "testdata/dnn_gender_classifier_v1.dat"
const age = "testdata/dnn_age_predictor_v1.dat"

// This example shows the basic usage of the package: create an
// recognizer, recognize faces, classify them using few known ones.
func Example_basic() {
	// Init the recognizer.
	rec, err := face.NewRecognizer()
	if err != nil {
		log.Fatalf("Can't init face recognizer: %v", err)
	}
	// Free the resources when you're finished.
	defer rec.Close()

	rec.SetCNNModel(cnn)
	rec.SetDescriptorModel(descr)
	rec.SetShapeModel(shape)
	rec.SetGenderModel(gender)
	rec.SetAgeModel(age)

	rec.SetSize(150)
	rec.SetPadding(0.25)
	rec.SetMinImageSize(100)
	rec.SetJittering(0)

	// Test image with 10 faces.
	testImagePristin := filepath.Join(dataDir, "pristin.jpg")
	// Recognize faces on that image.
	faces, err := rec.DetectFromFile(testImagePristin)
	if err != nil {
		log.Fatalf("Can't recognize: %v", err)
	}
	if len(faces) != 10 {
		log.Fatalf("Wrong number of faces")
	}

	// Fill known samples. In the real world you would use a lot of images
	// for each person to get better classification results but in our
	// example we just get them from one big image.
	var samples []face.Descriptor
	var cats []int32
	for i, f := range faces {
		rec.Recognize(&f)
		samples = append(samples, f.Descriptor)
		// Each face is unique on that image so goes to its own category.
		cats = append(cats, int32(i))
	}
	// Name the categories, i.e. people on the image.
	labels := []string{
		"Sungyeon", "Yehana", "Roa", "Eunwoo", "Xiyeon",
		"Kyulkyung", "Nayoung", "Rena", "Kyla", "Yuha",
	}
	// Pass samples to the recognizer.
	rec.SetSamples(samples, cats)

	// Now let's try to classify some not yet known image.
	testImageNayoung := filepath.Join(dataDir, "nayoung.jpg")
	nayoungFace, err := rec.DetectFromFile(testImageNayoung)
	if err != nil {
		log.Fatalf("Can't recognize: %v", err)
	}
	if len(faces) != 1 {
		log.Fatalf("Wrong number of faces")
	}
	rec.Recognize(&nayoungFace[0])

	catID := rec.Classify(nayoungFace[0].Descriptor)
	if catID < 0 {
		log.Fatalf("Can't classify")
	}
	// Finally print the classified label. It should be "Nayoung".
	fmt.Println(labels[catID])
}
