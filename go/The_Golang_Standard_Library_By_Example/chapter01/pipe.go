package main

import (
	"errors"
	"fmt"
	"io"
	"time"
)

func main() {
	pipeReader, pipeWriter := io.Pipe()
	go PipeWrite(pipeWriter)
	go PipeRead(pipeReader)
	time.Sleep(30 * time.Second)
}

func PipeRead(r *io.PipeReader) {
	buf := make([]byte, 128)
	for {
		fmt.Println("接收端开始阻塞5秒钟...")
		time.Sleep(5 * time.Second)
		fmt.Println("接收端开始接收")
		n, err := r.Read(buf)
		if err != nil {
			fmt.Println(err)
		}
		fmt.Printf("收到字节: %d\n buf内容: %s\n", n, buf[:n])
	}
}

func PipeWrite(w *io.PipeWriter) {
	data := []byte("The Golang Programming Language")
	for i := 0; i < 3; i++ {
		n, err := w.Write(data)
		if err != nil {
			fmt.Println(err)
			return
		}
		fmt.Printf("写入字节 %d\n", n)
	}
	w.CloseWithError(errors.New("写入端已关闭"))
}
