package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
)

func ReadFrom(reader io.Reader, num int) ([]byte, error) {
	p := make([]byte, num)
	n, err := reader.Read(p)
	if err != nil {
		return p[:n], err
	}
	return p, nil
}

func main() {
	// data,err := ReadFrom(os.Stdin,11)
	// data,err := ReadFrom(file,9)
	data, err := ReadFrom(strings.NewReader("from string"), 9)
	if err != nil && err != io.EOF {
		log.Fatal(err)
	} else {
		fmt.Println(string(data))
	}

	reader := strings.NewReader("Go Programming Language")
	p := make([]byte, 11)
	n, err := reader.ReadAt(p, 3)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s, %d\n", p, n)

	p = make([]byte, 100)
	n, err = reader.ReadAt(p, 3)
	if err != nil {
		fmt.Printf("%+v\n", err)
	}

	testWriteAt := func() {
		file, err := os.Create("/tmp/writeAt.txt")
		if err != nil {
			log.Fatal(err)
		}

		defer file.Close()
		file.WriteString("Hello, C++ Programming Language")
		n, err := file.WriteAt([]byte("Go "), 7)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(n)
	}
	testWriteAt()

	testReadFrom := func() {
		file, err := os.Open("/tmp/writeAt.txt")
		if err != nil {
			log.Fatal(err)
		}
		defer file.Close()
		writer := bufio.NewWriter(os.Stdout)
		writer.ReadFrom(file)
		writer.Flush()
	}

	testReadFrom()

	testWriteTo := func() {
		reader := strings.NewReader("Hello, WriteTo function!")
		reader.WriteTo(os.Stdout)
	}
	testWriteTo()

	testLimitedReader := func() {
		content := "Hello, Golang Programming Language"
		reader := strings.NewReader(content)
		limitReader := &io.LimitedReader{R: reader, N: 8}
		for limitReader.N > 0 {
			p := make([]byte, 2)
			limitReader.Read(p)
			fmt.Printf("%s\n", p)
		}
	}
	testLimitedReader()
}
