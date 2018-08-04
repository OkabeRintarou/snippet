package main

import (
	"fmt"
	"time"
)

func main() {
	t0 := time.Time{}
	fmt.Println(t0.IsZero())

	now := time.Now()
	fmt.Println(now.Unix())
	fmt.Println(now.UnixNano())
	fmt.Println(now.Location())

	/*
		t,err := time.Parse("2016-01-02 15:04:05","2016-06-13 09:14:00")
		if err != nil {
			panic(err)
		}
		fmt.Println(t)
		fmt.Println(time.Now().Sub(t).Hours())
	*/
}
