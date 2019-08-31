# go-face [![Build Status](https://travis-ci.org/Kagami/go-face.svg?branch=master)](https://travis-ci.org/Kagami/go-face) [![GoDoc](https://godoc.org/github.com/Kagami/go-face?status.svg)](https://godoc.org/github.com/Kagami/go-face)

go-face implements face recognition for Go using [dlib](http://dlib.net), a
popular machine learning toolkit. Read
[Face recognition with Go](https://hackernoon.com/face-recognition-with-go-676a555b8a7e)
article for some background details if you're new to
[FaceNet](https://arxiv.org/abs/1503.03832) concept.

## Requirements

To compile go-face you need to have dlib (>= 19.10) and libjpeg development
packages installed.

### Ubuntu 16.04, Ubuntu 18.04

You may use [dlib PPA](https://launchpad.net/~kagamih/+archive/ubuntu/dlib)
which contains latest dlib package compiled with Intel MKL support:

```bash
sudo add-apt-repository ppa:kagamih/dlib
sudo apt-get update
sudo apt-get install libdlib-dev libjpeg-turbo8-dev
```

### Ubuntu 18.10+, Debian sid

Latest versions of Ubuntu and Debian provide suitable dlib package so just run:

```bash
# Ubuntu
sudo apt-get install libdlib-dev libopenblas-dev libjpeg-turbo8-dev
# Debian
sudo apt-get install libdlib-dev libopenblas-dev libjpeg62-turbo-dev
```

**ONLY FOR UBUNTU 18.10+ AND DEBIAN SID:**  
It won't install pkgconfig metadata file so create one in
`/usr/local/lib/pkgconfig/dlib-1.pc` with the following content:

```
libdir=/usr/lib/x86_64-linux-gnu
includedir=/usr/include

Name: dlib
Description: Numerical and networking C++ library
Version: 19.10.0
Libs: -L${libdir} -ldlib -lblas -llapack
Cflags: -I${includedir}
Requires:
```

### macOS

Make sure you have [Homebrew](https://brew.sh) installed.

```bash
brew install pkg-config dlib
sed -i '' 's/^Libs: .*/& -lblas -llapack/' /usr/local/lib/pkgconfig/dlib-1.pc
```

### Windows

Make sure you have [MSYS2](https://www.msys2.org) installed.

1. Run `MSYS2 MSYS` shell from Start menu
2. Run `pacman -Syu` and if it asks you to close the shell do that
3. Run `pacman -Syu` again
4. Run `pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-dlib mingw-w64-x86_64-pkg-config`
5.
   1. If you already have Go and Git installed and available in PATH uncomment
      `set MSYS2_PATH_TYPE=inherit` line in `msys2_shell.cmd` located in MSYS2
      installation folder
   2. Otherwise run `pacman -S mingw-w64-x86_64-go git`
6. Run `MSYS2 MinGW 64-bit` shell from Start menu to compile and use go-face

### Other systems

Try to install dlib/libjpeg with package manager of your distribution or
[compile from sources](http://dlib.net/compile.html). Note that go-face won't
work with old packages of dlib such as libdlib18. Alternatively create issue
with the name of your system and someone might help you with the installation
process.

## Models

Currently `shape_predictor_5_face_landmarks.dat`, `mmod_human_face_detector.dat` and
`dlib_face_recognition_resnet_model_v1.dat` are required. You may download them
from [dlib-models](https://github.com/davisking/dlib-models) repo:

```bash
mkdir models && cd models
wget https://github.com/davisking/dlib-models/raw/master/shape_predictor_5_face_landmarks.dat.bz2
bunzip2 shape_predictor_5_face_landmarks.dat.bz2
wget https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2
bunzip2 dlib_face_recognition_resnet_model_v1.dat.bz2
wget https://github.com/davisking/dlib-models/raw/master/mmod_human_face_detector.dat.bz2
bunzip2 mmod_human_face_detector.dat.bz2
```

## Usage

To use go-face in your Go code:

```go
import "github.com/Kagami/go-face"
```

To install go-face in your $GOPATH:

```bash
go get github.com/Kagami/go-face
```

For further details see [GoDoc documentation](https://godoc.org/github.com/Kagami/go-face).

## Example

```go
package main

import (
	"fmt"
	"log"
	"path/filepath"

	"github.com/Kagami/go-face"
)

// Path to directory with models and test images. Here it's assumed it
// points to the <https://github.com/Kagami/go-face-testdata> clone.
const dataDir = "testdata"

// This example shows the basic usage of the package: create an
// recognizer, recognize faces, classify them using few known ones.
func main() {
	// Init the recognizer.
	rec, err := face.NewRecognizer(dataDir)
	if err != nil {
		log.Fatalf("Can't init face recognizer: %v", err)
	}
	// Free the resources when you're finished.
	defer rec.Close()

	// Test image with 10 faces.
	testImagePristin := filepath.Join(dataDir, "pristin.jpg")
	// Recognize faces on that image.
	faces, err := rec.RecognizeFile(testImagePristin)
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
	nayoungFace, err := rec.RecognizeSingleFile(testImageNayoung)
	if err != nil {
		log.Fatalf("Can't recognize: %v", err)
	}
	if nayoungFace == nil {
		log.Fatalf("Not a single face on the image")
	}
	catID := rec.Classify(nayoungFace.Descriptor)
	if catID < 0 {
		log.Fatalf("Can't classify")
	}
	// Finally print the classified label. It should be "Nayoung".
	fmt.Println(labels[catID])
}
```

Run with:

```bash
mkdir -p ~/go && cd ~/go  # Or cd to your $GOPATH
mkdir -p src/go-face-example && cd src/go-face-example
git clone https://github.com/Kagami/go-face-testdata testdata
edit main.go  # Paste example code
go get && go run main.go
```

## Test

To fetch test data and run tests:

```bash
make test
```

## FAQ

### How to improve recognition accuracy

There are few suggestions:

* Try CNN recognizing
* Try different tolerance values of `ClassifyThreshold`
* Try different size/padding/jittering values of `NewRecognizerWithConfig`
* Implement better classify heuristics (see [classify.cc](classify.cc))
* [Train](https://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html) network (`dlib_face_recognition_resnet_model_v1.dat`) on your own test data

## License

go-face is licensed under [CC0](COPYING).
