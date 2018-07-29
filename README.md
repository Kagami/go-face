# go-face [![Build Status](https://travis-ci.org/Kagami/go-face.svg?branch=master)](https://travis-ci.org/Kagami/go-face)

go-face implements face recognition for Go, using modern machine learning
toolkit [dlib](http://dlib.net) underneath.

## Dependencies

To compile go-face you need to have dlib (>= 19.10) and libjpeg development
packages installed.

### Ubuntu 16.04

You may use [dlib PPA](https://launchpad.net/~kagamih/+archive/ubuntu/dlib)
which contains latest dlib package compiled with Intel MKL support:

```bash
sudo add-apt-repository ppa:kagamih/dlib
sudo apt-get update
sudo apt-get install libdlib-dev libjpeg-turbo8-dev
```

If you're using other version of Ubuntu plese create issue and I may try to
make package for it too.

### Debian sid

Unstable branch of Debian contains suitable version of dlib so just run:

```bash
sudo apt-get install libdlib-dev libblas-dev liblapack-dev libjpeg62-turbo-dev
```

Debian's libdlib-dev doesn't provide pkgconfig metadata file so create one in
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

### Other

Try to install dlib/libjpeg with package manager of your distribution or
[compile from sources](http://dlib.net/compile.html).

Note that go-face won't work with old packages of dlib such as libdlib18.

## Usage


## Test

To fetch test data and run tests:

```bash
make test
```

## License

[CC0](COPYING).
