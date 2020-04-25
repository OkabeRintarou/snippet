package main

import (
	"io"
	"log"
	"net"
)

func main() {
	listener, err := net.Listen("tcp", ":8080")
	if err != nil {
		log.Fatalln("Unable to bind to port")
	}

	for {
		conn, err := listener.Accept()
		if err != nil {
			log.Fatalln("Unable to accept connection")
		}
		go handle(conn)
	}
}

func handle(src net.Conn) {
	dst, err := net.Dial("tcp", "example.com:80")
	if err != nil {
		log.Fatalln("Unable to connect to our unreachable host")
	}
	defer dst.Close()

	// Run in goroutine to prevent io.Copy from blocking
	go func() {
		if _, err = io.Copy(dst, src); err != nil {
			log.Fatalln(err)
		}
	}()

	if _, err = io.Copy(src, dst); err != nil {
		log.Fatalln(err)
	}
}
