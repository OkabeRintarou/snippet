package main

import (
	"fmt"
	"log"
	"os"
)

type FooReader struct {}
type FooWriter struct {}

func (r *FooReader) Read(b []byte) (n int, err error) {
	fmt.Print("in >")
	return os.Stdin.Read(b)
}

func (w *FooWriter) Write(b []byte) (n int, err error) {
	fmt.Print("out >")
	return os.Stdout.Write(b)
}

func main() {
	var (
		reader FooReader
		writer FooWriter
	)

	input := make([]byte, 4096)
	s, err := reader.Read(input)
	if err != nil {
		log.Fatalln("Unable to read data")
	}
	fmt.Printf("Read %d bytes from stdin\n", s)

	s, err = writer.Write(input)
	if err != nil {
		log.Fatalln("Unable to write data")
	}
	fmt.Printf("Write %d bytes to stdout\n", s)
}