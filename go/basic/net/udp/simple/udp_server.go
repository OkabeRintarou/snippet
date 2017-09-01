package main

import (
	"fmt"
	"log"
	"net"
)

func main() {
	listener,err := net.ListenUDP("udp",&net.UDPAddr{IP:net.ParseIP("127.0.0.1"),Port:6881})
	if err != nil {
		log.Fatal(err)
	}
	defer listener.Close()

	fmt.Printf("Local: <%s>\n",listener.LocalAddr().String())

	data := make([]byte,1024)
	for {
		n,remoteAddr,err := listener.ReadFromUDP(data)
		if err != nil {
			fmt.Printf("error during read: %s\n",err)
			continue
		}
		fmt.Printf("<%s> %s\n",remoteAddr,data[:n])
		_,err = listener.WriteToUDP([]byte("world"),remoteAddr)
		if err != nil {
			fmt.Printf(err.Error())
		}
	}
}
