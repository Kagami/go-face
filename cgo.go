// +build !arm64,!darwin

package face

// #cgo CXXFLAGS: -std=c++1z -Wall -O3 -DNDEBUG -march=native
import "C"
