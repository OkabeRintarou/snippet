package main

import (
	"fmt"
	"os"
	"syscall"
	"time"
)

func main() {
	_, err := os.Open("file.go")
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("os.Stdin fd = ", os.Stdin.Fd())
	fmt.Println("os.Stdout fd = ", os.Stdout.Fd())
	fmt.Println("os.Stderr fd = ", os.Stderr.Fd())

	file, err := os.OpenFile("/tmp/temp-xxx.txt", os.O_RDWR|os.O_CREATE|os.O_TRUNC, os.ModePerm)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	fs, err := os.Stat("/tmp/temp-xxx.txt")
	if err != nil {
		panic(err)
	}
	stat := fs.Sys().(*syscall.Stat_t)
	fmt.Println(time.Unix(stat.Atim.Unix()))

	// Sticky bit
	file, err = os.Create("temp.txt")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	defer os.Remove("temp.txt")

	fileMode := getFileMode(file)
	fmt.Println("file mode: ", fileMode)
	file.Chmod(fileMode | os.ModeSticky)
	fmt.Println("changed after, file mode: ", getFileMode(file))
}

func getFileMode(file *os.File) os.FileMode {
	info, err := file.Stat()
	if err != nil {
		panic(err)
	}
	return info.Mode()
}
