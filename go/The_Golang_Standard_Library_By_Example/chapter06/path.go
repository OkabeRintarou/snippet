package main

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
)

func main() {
	// Dir,Base,Ext
	p := "/home/syl/path.go"
	fmt.Println("Dir(path) = ", path.Dir(p))
	fmt.Println("Base(path) = ", path.Base(p))
	fmt.Println("Ext(path) = ", path.Ext(p))
	fmt.Printf("IsAbs(%s) ? %t\n", p, path.IsAbs(p))
	//Split
	dir, file := filepath.Split("/home/syl/studygolang")
	fmt.Println("Dir:", dir, ",File:", file)
	fmt.Println(`filepath.Join("/home/syl","study") = `, filepath.Join("/home/syl", "study"))
	fmt.Println("PATH = ", os.Getenv("PATH"))
	fmt.Println("On Unix:", filepath.SplitList(os.Getenv("PATH")))

	matches, err := filepath.Glob("/usr/lib/go/src/bufio/*.go")
	if err != nil {
		panic(err)
	}
	for _, m := range matches {
		fmt.Printf("%s ", m)
	}
	fmt.Println()

	filepath.Walk("../../..", func(path string, info os.FileInfo, err error) error {
		if info.IsDir() {
			return nil
		}
		fmt.Println("file:", info.Name(), "in directory:", path)
		return nil
	})
}
