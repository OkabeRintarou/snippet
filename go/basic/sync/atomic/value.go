package main

import (
	"fmt"
	"time"
	"sync/atomic"
	"math/rand"
	)

func loadConfig() string {
	return fmt.Sprintf("%d",rand.Int())
}

func main() {

	ch := make(chan struct{})

	var config atomic.Value
	config.Store(loadConfig())

	go func() {
		for {
			time.Sleep(1 * time.Second)
			config.Store(loadConfig())
		}
	}()

	for i := 0; i < 10; i++ {
		go func() {
			for {
				c := config.Load()
				fmt.Printf("Handle request using config %v\n",c)
				time.Sleep(2 * time.Second)
			}
		}()
	}

	<-ch
}
