package main

import (
	"encoding/gob"
	"fmt"
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
	var person Person
	loadGob("/tmp/person.gob", &person)
	fmt.Println("Person", person.String())
}

func loadGob(filename string, key interface{}) {
	file, err := os.Open(filename)
	checkError(err)

	defer file.Close()
	decoder := gob.NewDecoder(file)
	err = decoder.Decode(key)
	checkError(err)
}

func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s\n", err.Error())
		os.Exit(1)
	}
}
