export GOPATH = $(PWD)

testdata:
	git clone https://github.com/Kagami/go-face-testdata testdata

gofmt-staged:
	./gofmt-staged.sh

test: testdata gofmt-staged
	go test -v
