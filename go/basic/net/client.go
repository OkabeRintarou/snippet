package main

import (
		"fmt"
		"os"
		"net"
		"bufio"
	   )

func main(){
	conn,err := net.Dial("tcp","baidu.com:80")
	checkError(err)

	fmt.Fprintf(conn,"GET / HTTP/1.0\r\n\r\n")
	status,err := bufio.NewReader(conn).ReadString('\n')
	checkError(err)
	fmt.Printf("%s\n",status)
}

func checkError(err error){
	if err != nil{
		fmt.Fprintf(os.Stderr,"%s\n",err)
		os.Exit(-1)
	}
}
