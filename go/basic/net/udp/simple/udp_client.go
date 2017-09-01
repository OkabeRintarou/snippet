package main

import (
	"fmt"
	"log"
	"net"
)

func main() {
	sip := net.ParseIP("127.0.0.1")
	srcAddr := &net.UDPAddr{IP:net.IPv4zero,Port:0}
	dstAddr := &net.UDPAddr{IP:sip,Port:6881}

	conn,err := net.DialUDP("udp",srcAddr,dstAddr)
	if err != nil {
		log.Fatal(err)
	}
	defer conn.Close()
	conn.Write([]byte("Hello"))

	fmt.Printf("<%s>\n",conn.RemoteAddr())
}
