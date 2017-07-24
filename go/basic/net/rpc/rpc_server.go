package main

import (
	"log"
	"net"
	"net/http"
	"net/rpc"
	"time"
	)

type Hello struct {}

func (s *Hello) Prefix(in *string,out *string) error {
	*out = "hello " + *in
	time.Sleep(time.Second * 5)
	return nil
}

func main() {
	rpc.Register(new(Hello))
	rpc.HandleHTTP()
	listener,err := net.Listen("tcp",":8080")
	if err != nil {
		log.Fatal("Error listen:",err)
	}
	http.Serve(listener,nil)
}
