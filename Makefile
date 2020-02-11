.ONESHELL:
.PHONY: all download build sudo_install precommit gofmt-staged testdata

# Should contain src/github.com/Kagami/go-face or tests won't work
# properly (multiple definition of C functions).

# Temporary directory to put files into.
TMP_DIR?=/tmp/

BRANCH?=19.19

export GOPATH = $(PWD)/../../../..

all: download build sudo_install test clean
precommit: gofmt-staged

gofmt-staged:
	./gofmt-staged.sh

testdata:
	git clone https://github.com/Kagami/go-face-testdata testdata

download:
	curl -Lo $(TMP_DIR)dlib.tar.gz https://github.com/davisking/dlib/archive/v${BRANCH}.tar.gz
	cd $(TMP_DIR)
	tar xf dlib.tar.gz
	rm -rf dlib.tar.gz

build:
	cd $(TMP_DIR)dlib-${BRANCH}
	mkdir build
	cd build
	cmake -DCMAKE_BUILD_TYPE=Release -DDLIB_JPEG_SUPPORT=ON -DBUILD_SHARED_LIBS=YES -DDLIB_USE_BLAS=ON  -DDLIB_USE_LAPACK=ON  ..
	cmake --build . --config Release -- -j $(nproc --all)

sudo_install:
	cd $(TMP_DIR)dlib-${BRANCH}/build
	sudo make install

clean:
	rm -rf $(TMP_DIR)dlib-${BRANCH}
	rm -rf testdata

test: testdata
	go test -v