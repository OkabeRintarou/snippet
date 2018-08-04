package main

import (
	"fmt"
	"time"
)

func main() {
	c := make(chan struct{})
	go func() {
		defer func() {
			if recover() != nil {
				fmt.Println("Send on closed channel")
			}
		}()
		time.Sleep(3 * time.Second)
		c <- struct{}{}
	}()

	select {
	case <-c:
		fmt.Println("channel...")
	case <-time.After(2 * time.Second):
		close(c)
		fmt.Println("timeout...")
	}

	func() {
		t := time.Tick(time.Second)
		for _ = range t {
			fmt.Println("tick...")
		}
	}()
}
