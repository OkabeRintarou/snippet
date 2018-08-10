package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net"
)

func main() {
	sip := net.ParseIP("127.0.0.1")
	srcAddr := &net.UDPAddr{IP: net.IPv4zero, Port: 0}
	dstAddr := &net.UDPAddr{IP: sip, Port: 6881}

	conn, err := net.DialUDP("udp", srcAddr, dstAddr)
	if err != nil {
		log.Fatal(err)
	}
	defer conn.Close()

	for {
		var length int
		fmt.Scanf("%d", &length)
		msg := NewData(randomBytes(length))
		bytes, err := json.Marshal(msg)
		if err != nil {
			panic(err)
		}
		fmt.Printf("Send message length: %d\n", len(bytes))
		_, err = conn.Write(bytes)
		if err != nil {
			fmt.Printf("conn.Write error: %s\n", err.Error())
		}
	}
}
