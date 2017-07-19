package main

//
// Pileline构造原则
// * stage在发送完所有数据后关闭outbound channel
// * stage从发送方inbound channel读取完所有数据或者接收方广播了"done"消息后结束
//
import (
	"fmt"
	"sync"
	)

func gen(done <-chan struct{},nums ...int) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		for _,n := range nums {
			select {
			case out <- n:
			case <-done:
			}
		}
	}()
	return out
}

func sq(done <-chan struct{},in <-chan int) <-chan int {
	out := make(chan int)

	go func() {
		defer close(out)
		for n := range in {
			select {
			case out <- n * n:
			case <-done:
			}
		}
	}()
	return out
}

func merge(done <-chan struct{},cs ...<-chan int) <-chan int {
	var wg sync.WaitGroup
	out := make(chan int)

	output := func(c <-chan int) {
		defer wg.Done()
		// 循环直到读完上游的数据或者下游pipeline stage广播了结束消息
		for n := range c {
			select {
			case out <- n:
			case <-done:
				return
			}
		}
	}

	wg.Add(len(cs))
	for _,c := range cs {
		go output(c)
	}

	go func() {
		wg.Wait()
		close(out)
	}()

	return out
}

func main() {
	done := make(chan struct{}) // done用于向上游的pipeline stage广播
	defer close(done)
	out := merge(done,sq(done,gen(done,1)),sq(done,gen(done,2)))
	fmt.Println(<-out)
}

