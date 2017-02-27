package main

import (
		"os"
		"fmt"
		"net"
		)


func main(){
	ln,err := net.Listen("tcp",":9090")
	checkError(err)
	for{
		conn,err := ln.Accept()
		if err != nil{
			continue
		}
		go run(conn)
	}
}

func checkError(err error){
	if err != nil{
		fmt.Fprintf(os.Stderr,"%s\n",err)
		os.Exit(-1)
	}
}

func run(conn net.Conn){
	conn.Write([]byte("hello,world!"))
}
