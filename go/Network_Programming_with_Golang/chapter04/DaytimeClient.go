package main

import (
	"encoding/asn1"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"time"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s host:port\n", os.Args[0])
		os.Exit(1)
	}

	conn, err := net.Dial("tcp", os.Args[1])
	if err != nil {
		panic(err)
	}

	defer conn.Close()

	result, err := ioutil.ReadAll(conn)
	checkError(err)

	var newTime time.Time
	_, err = asn1.Unmarshal(result, &newTime)
	checkError(err)

	fmt.Println("After marshal/unmarshal: ", newTime)

}

func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s\n", err.Error())
		os.Exit(1)
	}
}
