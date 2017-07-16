package main

import (
	"io"
	"io/ioutil"
	"os"
	"log"
	"fmt"
	"strings"
)

func ReadFrom(reader io.Reader,num int) ([]byte,error) {
	p := make([]byte,num)
	n,err := reader.Read(p)
	if n > 0 {
		return p[:n],nil
	} else {
		return p,err
	}
}

func main(){
	// func Copy(dst Writer,src Reader)(written int64,err error)
	// @@ 将src中的内容拷贝到dst直到遇到EOF或者error
	reader := strings.NewReader("hello,world")
	if _,err := io.Copy(os.Stdout,reader);err != nil{
		log.Fatal(err)
	}

	fmt.Println()
	// func CopyN(dst Writer,src Reader,n int64)(written int64,err error)
	// @@ 将src中的内容拷贝n个字节到dst中直到遇到错误
	reader = strings.NewReader("hello,world")
	if _,err := io.CopyN(os.Stdout,reader,5);err != nil{
		log.Fatal(err)
	}
	fmt.Println()

	// func ReadAtLeast(r Reader,buf []byte,min int)(n int err error)
	// @@ ReadAtLeast 将 r 读取到 buf 中,直到读了至少 min 个字节
	// @@ 若没有读取到字节错误就只是EOF
	// @@ 如果发生EOF并且读到的字节数小于 min,则返回ErrUnexpectedEOF 错误
	// @@ 如果 min > len(buf),返回错误 ErrShortBUffer
	// @@ 当且仅当 err == nil时才有 n >= min
	buf := make([]byte,5)
	reader = strings.NewReader("hello,world")
	if _,err := io.ReadAtLeast(reader,buf,5);err != nil{
		log.Fatal(err)
	}
	fmt.Println(string(buf))

	// func ReadFull(r Reader,buf []byte)(n int err error)
	// @@ ReadFull 精确地从 r 中将 len(buf) 个字节读取到 buf 中.
	// @@ 如果没有读取到字节,错误是 EOF
	// @@ 如果一个 EOF发生但没有读满 buf 则返回 ErrUnexpectedEOF 错误
	buf = make([]byte,11)
	reader = strings.NewReader("hello,world")
	if _,err := io.ReadFull(reader,buf); err != nil{
		log.Fatal(err)
	}
	fmt.Println(string(buf))

	// func WriteString(w Writer,s string)(n int err error)
	w,err := ioutil.TempFile("/tmp","tmp")
	if err != nil{
		log.Fatal(err)
	}
	n,err := io.WriteString(w,"hello,world")
	if err != nil{
		log.Fatal(err)
	}
	fmt.Printf("%d bytes writted\n",n);
}
