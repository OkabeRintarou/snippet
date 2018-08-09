package main

import (
	"encoding/json"
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

func main() {
	person := Person{
		Name: Name{Family: "Newmarch", Personal: "Jan"},
		Email: []Email{
			Email{Kind: "home", Address: "jan@newmarch.com"},
			Email{Kind: "work", Address: "j.newmarch@boxhill.edu.au"},
		},
	}

	saveJSON("/tmp/person.json", person)
}

func saveJSON(filename string, key interface{}) {
	outFile, err := os.Create(filename)
	checkError(err)

	defer outFile.Close()
	encoder := json.NewEncoder(outFile)
	err = encoder.Encode(key)
	checkError(err)
}

func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s\n", err.Error())
		os.Exit(1)
	}
}
