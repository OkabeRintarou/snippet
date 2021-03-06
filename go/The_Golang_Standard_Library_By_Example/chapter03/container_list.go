package main

import (
	"container/list"
	"fmt"
)

func main() {
	list := list.New()
	list.PushBack(1)
	list.PushBack(2)

	fmt.Printf("len:%v\n", list.Len())
	fmt.Printf("first:%#v\n", list.Front())
	fmt.Printf("second:%#v\n", list.Front().Next())

	list.Init()
	fmt.Printf("len:%v\n", list.Len())
	if list.Front() == nil {
		fmt.Println("Init clear the list")
	}
}
