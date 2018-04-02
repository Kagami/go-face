export GOPATH = $(PWD)

gofmt-staged:
	./gofmt-staged.sh

test: gofmt-staged
	go test -v
