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

	log.Println("Begin asynchronous rpc")
	start := time.Now()
	call := client.Go("Hello.Prefix",&in,&out,nil)
	log.Println("End asynchronous rpc,spend time ",time.Now().Sub(start))
	reply := <-call.Done
	log.Println("Fetch data spend time ",time.Now().Sub(start))
	checkError(reply.Error,"asynchronous rpc")
	log.Println(out)
}

func checkError(err error,msg string) {
	if err != nil {
		log.Fatal("Error " + msg + ":" + err.Error())
	}
}
