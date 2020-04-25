package main

import (
	"fmt"
	"net"
)

func main() {
	for i := 0; i < 1024; i++ {
		addr := fmt.Sprintf("qq.com:%d", i)
		conn, err := net.Dial("tcp", addr)
		if err != nil {
			// port is closed or filtered
			continue
		}
		conn.Close()
		fmt.Printf("%d open\n", i)
	}
}
