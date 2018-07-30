package main

import (
	"fmt"
	"strconv"
)

func main() {
	// ParseInt, ParseUint and Atoi
	func() {
		n, err := strconv.ParseInt("128", 10, 8)
		fmt.Printf("n = %d\terror:%s\n", n, err)
		v, err := strconv.Atoi("128")
		if err != nil {
			fmt.Println(err)
		}
		fmt.Printf("%d\n", v)
	}()

	// FormatInt, FormatUint and Itoa
	func() {
		v := strconv.FormatInt(12345, 8)
		fmt.Println(v)
		fmt.Println(strconv.FormatInt(12345, 16))
	}()

	// ParseBool, FormatBool and AppendBool

	// ParseFloat, FormatFloat and AppendFloat
	func() {
		fmt.Println(strconv.FormatFloat(1223.13252, 'e', 3, 32))
		fmt.Println(strconv.FormatFloat(1223.13252, 'g', 3, 32))
	}()

	s := strconv.Quote(`"Fran & Freddie's Diner	☺"`)
	fmt.Println(s)
	fmt.Println(strconv.Quote("Hello, gopher!"))

	s = strconv.QuoteRuneToASCII('☺')
	fmt.Println(s)
}
