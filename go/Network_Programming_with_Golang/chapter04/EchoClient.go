package main

import (
	"encoding/json"
	"fmt"
	"io"
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
	if len(os.Args) != 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s host:port\n", os.Args[0])
		os.Exit(1)
	}

	conn, err := net.Dial("tcp", os.Args[1])
	checkError(err)

	person := Person{
		Name: Name{Family: "Newmarch", Personal: "Jan"},
		Email: []Email{
			Email{Kind: "home", Address: "jan@newmarch.com"},
			Email{Kind: "work", Address: "j.newmarch@boxhill.edu.au"},
		},
	}

	decoder := json.NewDecoder(conn)
	encoder := json.NewEncoder(conn)

	for {
		err = encoder.Encode(person)
		if err != nil {
			if err == io.EOF {
				break
			} else {
				panic(err)
			}
		}
		var newPerson Person
		err = decoder.Decode(&newPerson)
		if err != nil {
			if err == io.EOF {
				break
			} else {
				panic(err)
			}
		} else {
			fmt.Println(newPerson.String())
		}

	}

}

func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s\n", err.Error())
		os.Exit(1)
	}
}
