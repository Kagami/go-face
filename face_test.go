package face_test

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"path/filepath"
	"strings"
	"testing"
	"unsafe"

	"github.com/Kagami/go-face"
)

var (
	rec *face.Recognizer

	idolTests = map[string]string{
		"elkie.jpg":      "Elkie, CLC",
		"chaeyoung.jpg":  "Chaeyoung, Twice",
		"chaeyoung2.jpg": "Chaeyoung, Twice",
		"sejeong.jpg":    "Sejeong, Gugudan",
		"jimin.jpg":      "Jimin, AOA",
		"jimin2.jpg":     "Jimin, AOA",
		"jimin4.jpg":     "Jimin, AOA",
		"meiqi.jpg":      "Mei Qi, WJSN",
		"chaeyeon.jpg":   "Chaeyeon, DIA",
		"chaeyeon3.jpg":  "Chaeyeon, DIA",
		"tzuyu2.jpg":     "Tzuyu, Twice",
		"nayoung.jpg":    "Nayoung, PRISTIN",
		"luda2.jpg":      "Luda, WJSN",
		"joy.jpg":        "Joy, Red Velvet",
	}
)

type Idol struct {
	ID       string `json:"id"`
	Name     string `json:"name"`
	BandName string `json:"band_name"`
}

type IdolFace struct {
	Descriptor string `json:"descriptor"`
	IdolID     string `json:"idol_id"`
}

type IdolData struct {
	Idols []Idol     `json:"idols"`
	Faces []IdolFace `json:"faces"`
	byID  map[string]*Idol
}

type TrainData struct {
	samples []face.Descriptor
	cats    []int32
	labels  []string
}

func getTPath(fname string) string {
	return filepath.Join("testdata", fname)
}

func getIdolData() (idata *IdolData, err error) {
	data, err := ioutil.ReadFile(getTPath("idols.json"))
	if err != nil {
		return
	}
	idata = &IdolData{}
	err = json.Unmarshal(data, idata)
	if err != nil {
		return
	}
	idata.byID = make(map[string]*Idol)
	for i, _ := range idata.Idols {
		idol := &idata.Idols[i]
		idata.byID[idol.ID] = idol
	}
	return
}

func str2descr(s string) face.Descriptor {
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		panic(err)
	}
	return *(*face.Descriptor)(unsafe.Pointer(&b[0]))
}

func getTrainData(idata *IdolData) (tdata *TrainData) {
	var samples []face.Descriptor
	var cats []int32
	var labels []string

	var catID int32
	var prevIdolID string
	catID = -1
	for i, _ := range idata.Faces {
		iface := &idata.Faces[i]
		descriptor := str2descr(iface.Descriptor)
		samples = append(samples, descriptor)
		if iface.IdolID != prevIdolID {
			catID++
			labels = append(labels, iface.IdolID)
		}
		cats = append(cats, catID)
		prevIdolID = iface.IdolID
	}

	tdata = &TrainData{
		samples: samples,
		cats:    cats,
		labels:  labels,
	}
	return
}

func recognizeAndClassify(fpath string, tolerance float32) (id int, err error) {
	id = -1
	faces, err := rec.DetectFromFile(fpath)
	if err != nil || len(faces) <= 0 {
		return
	}
	f := faces[0]
	rec.Recognize(&f)
	if tolerance < 0 {
		id = rec.Classify(f.Descriptor)
	} else {
		id = rec.ClassifyThreshold(f.Descriptor, tolerance)
	}
	return
}

/*func TestSerializationError(t *testing.T) {
	_, err := face.NewRecognizer()
	switch err.(type) {
	case face.SerializationError:
		// skip
	default:
		t.Fatalf("Wrong error: %v", err)
	}
}*/

func TestInit(t *testing.T) {
	var err error
	rec, err = face.NewRecognizer()
	if err != nil {
		t.Fatalf("Can't init face recognizer: %v", err)
	}
	rec.SetCNNModel(cnn)
	rec.SetDescriptorModel(descr)
	rec.SetShapeModel(shape)
}

func TestImageLoadError(t *testing.T) {
	_, err := rec.DetectFromBuffer([]byte{1, 2, 3})
	switch err.(type) {
	case face.ImageLoadError:
		// skip
	default:
		t.Fatalf("Wrong error: %v", err)
	}
}

func TestNumFaces(t *testing.T) {
	faces, err := rec.DetectFromFile(getTPath("pristin.jpg"))
	if err != nil {
		t.Fatalf("Can't get faces: %v", err)
	}
	numFaces := len(faces)
	if numFaces != 10 {
		t.Fatalf("Wrong number of faces: %d", numFaces)
	}
}

func TestEmptyClassify(t *testing.T) {
	var sample face.Descriptor
	id := rec.Classify(sample)
	if id >= 0 {
		t.Fatalf("Shouldn't recognize but got %d category", id)
	}
}

func TestIdols(t *testing.T) {
	idata, err := getIdolData()
	if err != nil {
		t.Fatalf("Can't get idol data: %v", err)
	}
	tdata := getTrainData(idata)
	rec.SetSamples(tdata.samples, tdata.cats)

	for fname, expected := range idolTests {
		t.Run(fname, func(t *testing.T) {
			names := strings.Split(expected, ", ")
			expectedIname := names[0]
			expectedBname := names[1]

			catID, err := recognizeAndClassify(getTPath(fname), -1)
			if err != nil {
				t.Fatalf("Can't recognize: %v", err)
			}
			if catID < 0 {
				t.Errorf("%s: expected “%s” but not recognized", fname, expected)
				return
			}
			idolID := tdata.labels[catID]
			idol := idata.byID[idolID]
			actualIname := idol.Name
			actualBname := idol.BandName

			if expectedIname != actualIname || expectedBname != actualBname {
				actual := fmt.Sprintf("%s, %s", actualIname, actualBname)
				t.Errorf("%s: expected “%s” but got “%s”", fname, expected, actual)
			}
		})
	}
}

func TestClassifyThreshold(t *testing.T) {
	id, err := recognizeAndClassify(getTPath("nana.jpg"), 0.1)
	if err != nil {
		t.Fatalf("Can't recognize: %v", err)
	}
	if id >= 0 {
		t.Fatalf("Shouldn't recognize but got %d category", id)
	}
	id, err = recognizeAndClassify(getTPath("nana.jpg"), 0.8)
	if err != nil {
		t.Fatalf("Can't recognize: %v", err)
	}
	if id < 0 {
		t.Fatalf("Should have recognized but got %d category", id)
	}
}

func TestClose(t *testing.T) {
	rec.Close()
}
