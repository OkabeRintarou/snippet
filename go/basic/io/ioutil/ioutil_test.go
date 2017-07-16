package main

import (
	"os"
	"io/ioutil"
	"strings"
	"testing"
)

func checkSize(t *testing.T,path string,size int64) {
	dir,err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat %q (looking for size %d): %s",path,size,err)
	}
	if dir.Size() != size {
		t.Errorf("Stat %q: size %d want %d",path,dir.Size(),size)
	}
}

// func ReadFile(filename string)([]byte,error)
// ReadFile 返回文件的全部内容或者错误
func TestReadFile(t *testing.T) {
	filename := "rumpelstilzchen"
	contents,err := ioutil.ReadFile(filename)
	if err == nil {
		t.Fatalf("ReadFile %s: error expected, none found",filename)
	}

	filename = "ioutil_test.go"
	contents,err = ioutil.ReadFile(filename)
	if err != nil {
		t.Fatalf("ReadFile %s: %v",filename,err)
	}

	checkSize(t,filename,int64(len(contents)))
}


func TestReadAll(t *testing.T) {
	reader := strings.NewReader("hello,world.")
	contents,err := ioutil.ReadAll(reader)
	if err != nil {
		t.Fatalf("ReadAll unexpected error")
	}
	if string(contents) != "hello,world." {
		t.Fatalf("RealAll want %s,actual %s","hello,world.",string(contents))
	}
}

func TestReadDir(t *testing.T) {
	_,err := ioutil.ReadDir(".")
	if err != nil {
		t.Fatalf("ReadDir unexpected error")
	}
}
