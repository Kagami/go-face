# go-face [![Build Status](https://travis-ci.org/Kagami/go-face.svg?branch=master)](https://travis-ci.org/Kagami/go-face)

go-face implements face recognition for Go using [dlib](http://dlib.net), a
popular machine learning toolkit.

## Requirements

To compile go-face you need to have dlib (>= 19.10) and libjpeg development
packages installed.

### Ubuntu 16.04, Ubuntu 18.04

You may use my [dlib PPA](https://launchpad.net/~kagamih/+archive/ubuntu/dlib)
which contains latest dlib package compiled with Intel MKL support:

```bash
sudo add-apt-repository ppa:kagamih/dlib
sudo apt-get update
sudo apt-get install libdlib-dev libjpeg-turbo8-dev
```

### Ubuntu 18.10+, Debian sid

Latest versions of Ubuntu and Debian provide suitable dlib package so just run:

```bash
sudo apt-get install libdlib-dev libblas-dev liblapack-dev libjpeg62-turbo-dev
```

Unfortunately libdlib-dev doesn't contain pkgconfig metadata file so create one
in `/usr/local/lib/pkgconfig/dlib-1.pc` with the following content:

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

### Other systems

Try to install dlib/libjpeg with package manager of your distribution or
[compile from sources](http://dlib.net/compile.html). Note that go-face won't
work with old packages of dlib such as libdlib18.

## Models

Currently `shape_predictor_5_face_landmarks.dat` and
`dlib_face_recognition_resnet_model_v1.dat` are required. You may download them
from [dlib-models](https://github.com/davisking/dlib-models) repo.

## Usage


## Test

To fetch test data and run tests:

```bash
make test
```

## License

[CC0](COPYING).
