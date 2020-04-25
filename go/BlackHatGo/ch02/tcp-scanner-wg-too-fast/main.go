package main

import (
	"fmt"
	"net"
	"sync"
)

func main() {
	var wg sync.WaitGroup

	for i := 0; i < 1024; i++ {
		wg.Add(1)
		go func (port int) {
			defer wg.Done()
			addr := fmt.Sprintf("qq.com:%d", port)
			conn, err := net.Dial("tcp", addr)
			if err != nil {
				// port is closed or filtered
				return
			}
			conn.Close()
			fmt.Printf("%d open\n", port)
		}(i)
	}

	wg.Wait()
}
