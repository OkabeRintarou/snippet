package main

import (
	"container/ring"
	"fmt"
)

func main() {
	ring := ring.New(3)
	for i := 1; i <= 3; i++ {
		ring.Value = i
		ring = ring.Next()
	}
	s := 0
	ring.Do(func(x interface{}) {
		s += x.(int)
	})
	fmt.Println(s)
}
