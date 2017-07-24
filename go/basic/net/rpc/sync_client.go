package main

import (
	"log"
	"time"
	"net/rpc"
	)

func main() {
	client,err := rpc.DialHTTP("tcp",":8080")
	checkError(err,"rpc.DialHTTP")
	defer client.Close()

	var out string
	in := "OkabeRintarou"

	// synchronous rpc
	log.Println("Begin client.Call")
	start := time.Now()
	err = client.Call("Hello.Prefix",&in,&out)
	log.Println("End client.Call,spend time ",time.Now().Sub(start))
	checkError(err,"synchronous rpc")
	log.Println(out)
}

func checkError(err error,msg string) {
	if err != nil {
		log.Fatal("Error " + msg + err.Error())
	}
}
