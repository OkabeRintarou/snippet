package main

import (
	"encoding/json"
	"fmt"
	"net"
	"os"
)

type Person struct {
	Name  Name
	Email []Email
}

type Name struct {
	Family   string
	Personal string
}

type Email struct {
	Kind    string
	Address string
}

func (p Person) String() string {
	s := p.Name.Personal + " " + p.Name.Family
	for _, v := range p.Email {
		s += "\n" + v.Kind + ": " + v.Address
	}
	return s
}

func main() {
	service := "0.0.0.0:1200"
	listener, err := net.Listen("tcp", service)
	checkError(err)

	defer listener.Close()

	for {
		conn, err := listener.Accept()
		if err != nil {
			continue
		}
		go handleClient(conn)
	}
}

func handleClient(conn net.Conn) {
	encoder := json.NewEncoder(conn)
	decoder := json.NewDecoder(conn)

	defer conn.Close()

	for n := 0; n < 10; n++ {
		var person Person
		decoder.Decode(&person)
		fmt.Println(person.String())
		encoder.Encode(person)
	}
}

func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s\n", err.Error())
		os.Exit(1)
	}
}
