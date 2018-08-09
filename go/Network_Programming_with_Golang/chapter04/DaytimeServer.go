package main

import (
	"encoding/asn1"
	"net"
	"time"
)

func main() {
	service := ":1200"
	listener, err := net.Listen("tcp", service)
	if err != nil {
		panic(err)
	}

	defer listener.Close()

	for {
		conn, err := listener.Accept()
		if err != nil {
			continue
		}

		go handleClient(conn)
	}
}

func handleClient(conn net.Conn) {
	defer conn.Close()
	daytime := time.Now()
	mdata, _ := asn1.Marshal(daytime)
	conn.Write(mdata)
}
