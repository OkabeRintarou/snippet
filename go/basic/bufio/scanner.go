package main

import (
	"bufio"
	"fmt"
	"os"
	)

func main() {
	file,err := os.Open("scanner.go")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		fmt.Println(scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr,"reading file:",err)
	}
}
