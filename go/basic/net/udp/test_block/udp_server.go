package main

import (
	"fmt"
	"log"
	"net"
	"time"
)

func main() {
	listener, err := net.ListenUDP("udp", &net.UDPAddr{IP: net.ParseIP("127.0.0.1"), Port: 6881})
	if err != nil {
		log.Fatal(err)
	}
	defer listener.Close()

	fmt.Printf("Local: <%s>\n", listener.LocalAddr().String())

	for {
		time.Sleep(time.Minute)
	}
}
