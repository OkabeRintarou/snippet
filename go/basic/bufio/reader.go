package main

import (
		"bufio"
		"strings"
		"fmt"
	)

func main() {
	fmt.Printf("ReadSlice:\n")
	reader := bufio.NewReader(strings.NewReader("Hello\nWorld"))
	line,_ := reader.ReadSlice('\n')
	fmt.Printf("The line:%s\n",line)
	n,_ := reader.ReadSlice('\n')
	fmt.Printf("The line:%s\n",line)
	fmt.Println(string(n))

	fmt.Printf("\nReadBytes:\n")
	reader = bufio.NewReader(strings.NewReader("Hello\nWorld"))
	line,_ = reader.ReadBytes('\n')
	fmt.Printf("The line:%s\n",line)
	n,_ = reader.ReadBytes('\n')
	fmt.Printf("The line:%s\n",line)
	fmt.Println(string(n))

	fmt.Printf("\nReadLine:\n")
	reader = bufio.NewReader(strings.NewReader("Hello\nWorld"))
	line,_,_ = reader.ReadLine()
	fmt.Printf("The line:%s\n",line)
	n,_,_ = reader.ReadLine()
	fmt.Printf("The line:%s\n",line)
	fmt.Println(string(n))
}
