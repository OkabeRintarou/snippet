package main

import (
	"encoding/asn1"
	"fmt"
	"os"
)

type T struct {
	Field1 int
	Field2 int
}

func main() {
	mdata, err := asn1.Marshal(13)
	checkError(err)

	var n int
	_, err1 := asn1.Unmarshal(mdata, &n)
	checkError(err1)

	fmt.Println("After marshal/unmarshal:", n)

	s := "hello"
	mdata, err = asn1.Marshal(s)
	checkError(err)

	var newstr string
	_, err1 = asn1.Unmarshal(mdata, &newstr)
	checkError(err1)

	fmt.Println("After marshal/unmarshal:", newstr)

	s = "hello  \u00bc"
	mdata, err = asn1.Marshal(s)
	if err != nil {
		fmt.Println(err)
	} else {
		_, err1 = asn1.Unmarshal(mdata, &newstr)
		if err1 != nil {
			fmt.Println(err1)
		} else {
			fmt.Println("After marshal/unmarshal:", newstr)
		}
	}

	testStruct()
}

func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s\n", err.Error())
		os.Exit(1)
	}
}

func testStruct() {
	// use variable
	var t1 T = T{100, 200}
	mdata1, _ := asn1.Marshal(t1)
	var newT1 T
	asn1.Unmarshal(mdata1, &newT1)
	fmt.Println("After marshal/unmarshal:", newT1)

	// using Pointer
	var t2 = new(T)
	*t2 = T{100, 200}
	mdata2, _ := asn1.Marshal(*t2)

	var newT2 = new(T)
	asn1.Unmarshal(mdata2, newT2)
	fmt.Println("After marshal/unmarshal:", newT2)

	var newT22 = new(T2)
	asn1.Unmarshal(mdata2, newT22)
	fmt.Println("After marshal/unmarshal:", newT22)
}

type T2 struct {
	F1 int
	F2 int
}
