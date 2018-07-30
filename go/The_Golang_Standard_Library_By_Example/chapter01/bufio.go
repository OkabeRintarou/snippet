package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

func main() {
	reader := bufio.NewReader(strings.NewReader("hello\nworld"))
	line, _ := reader.ReadSlice('\n')
	fmt.Printf("the line: %s\n", line)
	n, _ := reader.ReadSlice('\n')
	fmt.Printf("the line: %s\n", line)
	fmt.Println(string(n))

	reader = bufio.NewReaderSize(strings.NewReader("http://www.github.com/"), 16)
	line, err := reader.ReadSlice('\n')
	fmt.Printf("line: %s\terror: %s\n", line, err)
	line, err = reader.ReadSlice('\n')
	fmt.Printf("line: %s\terror: %s\n", line, err)

	reader = bufio.NewReaderSize(strings.NewReader("http:/www.github.com/\n"), 16)
	line, err = reader.ReadSlice('\n')
	fmt.Printf("line: %s\terror: %s\n", line, err)
	line, err = reader.ReadSlice('\n')
	fmt.Printf("line: %s\terror: %v\n", line, err)

	file, err := os.Open("./bufio.go")
	if err != nil {
		panic(err)
	}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		fmt.Println(scanner.Text())
	}

	const input = "This is The Golang Standard Library.\nWelcome you!"
	scanner = bufio.NewScanner(strings.NewReader(input))
	scanner.Split(bufio.ScanWords)
	count := 0
	for scanner.Scan() {
		count++
	}
	fmt.Printf("count: %d\n", count)

	testScanner := func() {
		file, err := os.Create("/tmp/create.txt")
		if err != nil {
			panic(err)
		}
		defer file.Close()
		file.WriteString("http://github.com/OkabeRintarou/snippet\nWelcome Gopher to visit my github page\nHello\n")
		file.Seek(0, os.SEEK_SET)
		scanner := bufio.NewScanner(file)
		scanner.Split(bufio.ScanWords)
		for scanner.Scan() {
			fmt.Println(scanner.Text())
		}
	}

	testScanner()
}
