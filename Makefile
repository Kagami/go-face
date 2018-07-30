# Should contain src/github.com/Kagami/go-face or tests won't work
# properly (multiple definition of C functions).
export GOPATH = $(PWD)/../../../..

all: test
precommit: gofmt-staged

gofmt-staged:
	./gofmt-staged.sh

testdata:
	git clone https://github.com/Kagami/go-face-testdata testdata

test: testdata
	go test -v
