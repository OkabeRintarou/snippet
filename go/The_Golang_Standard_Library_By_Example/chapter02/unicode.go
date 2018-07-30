package main

import (
	"fmt"
	"strconv"
	"unicode"
	"unicode/utf8"
)

func testUtf8() {
	valid := 'a'
	invalid := rune(0xfffffff)
	fmt.Printf("0x%x ValidRune? %t\n", valid, utf8.ValidRune(valid))
	fmt.Printf("0x%x ValidRune? %t\n", invalid, utf8.ValidRune(invalid))
	fmt.Println(utf8.RuneLen(rune('a')))
	fmt.Println(utf8.RuneLen('中'))
	fmt.Println(utf8.RuneCount([]byte{'a', 'b', 'c'}))
	fmt.Println(utf8.RuneCountInString("中国"))

	b := []byte("Hello, 世界")
	for len(b) > 0 {
		r, size := utf8.DecodeRune(b)
		fmt.Printf("%c %v\n", r, size)
		b = b[size:]
	}

	r := '世'
	buf := make([]byte, 3)
	n := utf8.EncodeRune(buf, r)
	fmt.Println(buf)
	fmt.Println(n)

	b = []byte("Hello, 世界")
	for len(b) > 0 {
		r, size := utf8.DecodeLastRune(b)
		fmt.Printf("%c %v\n", r, size)
		b = b[:len(b)-size]
	}
}

func testUtf16() {
}

func main() {
	single := '\u0015'
	fmt.Printf("%c is control? %t\n", single, unicode.IsControl(single))
	single = '\ufe35'
	fmt.Printf("%c is control? %t\n", single, unicode.IsControl(single))

	digit := rune('1')
	fmt.Printf("%s is digit? %t\n", strconv.QuoteRune(digit), unicode.IsDigit(digit))
	fmt.Printf("%s is number? %t\n", strconv.QuoteRune(digit), unicode.IsNumber(digit))

	letter := rune('Ⅷ')
	fmt.Printf("%s is digit? %t\n", strconv.QuoteRune(letter), unicode.IsDigit(letter))
	fmt.Printf("%s is number? %t\n", strconv.QuoteRune(letter), unicode.IsNumber(letter))

	testUtf8()
	testUtf16()
}
