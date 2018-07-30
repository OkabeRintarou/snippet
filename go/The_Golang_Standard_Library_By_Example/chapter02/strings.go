package main

import (
	"fmt"
	"os"
	"reflect"
	"strings"
	"unicode"
)

func main() {
	// Contains,ContainsAny and ContainsRune
	fmt.Println(strings.Contains("hello,world", "ello")) // true
	fmt.Println(strings.ContainsAny("failure", "i & u")) // true
	fmt.Println(strings.ContainsRune("中国", '中'))         // true

	fmt.Println(strings.Count("five", ""))         // 5
	fmt.Println(strings.Count("hello,world", "o")) // 2
	fmt.Println(strings.Count("fivevev", "vev"))   // 1

	// Fields and FieldsFunc
	func() {
		reflect.DeepEqual(strings.Fields(" foo bar baz "), []string{"foo", "bar", "baz"})
		reflect.DeepEqual(strings.FieldsFunc(" foo bar \n baz", unicode.IsSpace), []string{"foo", "bar", "baz"})
	}()

	// Split,SplitAfter,SplitN and SplitAfterN
	func() {
		reflect.DeepEqual(strings.Split("abc", ""), []string{"a", "b", "c"})
		// difference between Split and SplitAfter: SplitAfter retain the `seq` in the result array
		reflect.DeepEqual(strings.Split("foo,bar,baz", ","), []string{"foo", "bar", "baz"})
		reflect.DeepEqual(strings.SplitAfter("foo,bar,baz", "r"), []string{"foo,", "bar,", "baz"})
		reflect.DeepEqual(strings.SplitN("a,b,c", ",", 2), []string{"a", "b"})
		reflect.DeepEqual(strings.SplitN("a,b,c", ",", 0), nil)
	}()

	// HasPrefix and HasSuffix
	// func HasPrefix(s,prefix string) bool
	// func HasSuffix(s,suffix string) bool

	// Index, IndexAny, IndexFunc, IndexByte and IndexRune
	func() {
		fmt.Printf("%d\n", strings.IndexFunc("studygolang.com", func(c rune) bool {
			if c > 'u' {
				return true
			}
			return false
		})) // 4
		fmt.Printf("%d\n", strings.LastIndexAny("studygolang.com", "aeiou")) // 13
	}()

	// Join
	// func Join(a[] string, sep string) string
	fmt.Println(strings.Join([]string{"name=xxx", "age=xx"}, "&"))

	// Repeat
	// func Repeat(s string, count int) string
	fmt.Println(strings.Repeat("hello", 5))

	// Replace
	// func Replace(s, old, new string, n int) string
	fmt.Println(strings.Replace("oink oink oink", "k", "ky", 2))      // oinky oinky oink
	fmt.Println(strings.Replace("oink oink oink", "oink", "moo", -1)) // moo moo moo

	// Replacer
	func() {
		r := strings.NewReplacer("<", "&lt;", ">", "&gt;")
		fmt.Println(r.Replace("This is <b>HTML</b>"))
		r.WriteString(os.Stdout, "<title>Hello,Gopther!</title>")
	}()

	// EqualFold
	fmt.Println(strings.EqualFold("Go", "gO"))

	// Map
	// func Map(mapping func(rune) rune) string
	rot13 := func(r rune) rune {
		switch {
		case r >= 'A' && r <= 'Z':
			return 'A' + (r-'A'+13)%26
		case r >= 'a' && r <= 'z':
			return 'a' + (r-'a'+13)%26
		}
		return r
	}
	fmt.Println(strings.Map(rot13, "Twas brilling and the slithy gopher..."))

	// Title
	fmt.Println(strings.Title("her royal highness"))

	// ToLower,ToLowerSpecial,ToUpper,ToUpperSpecial
	fmt.Println(strings.ToLowerSpecial(unicode.TurkishCase, "Önnek İş"))

	// Trim
	// func Trim(s string, cutset string) string
	fmt.Println(strings.Trim("¡¡¡Hello, Gophers!!!", "!¡"))
	// TrimFunc
	fmt.Println(strings.TrimFunc("¡¡¡Hello, Gophers!!!", func(c rune) bool {
		return !unicode.IsLetter(c) && !unicode.IsNumber(c)
	}))
	// TrimLeft,TrimRight,TrimLeftFunc,TrimRightFunc
	// TrimSpace

	// TrimSuffix
	func() {
		var s = "¡¡¡Hello, Gophers!!!"
		fmt.Println(strings.TrimSuffix(s, ", Gophers!!!"))
		fmt.Println(strings.TrimSuffix(s, ", Marmots!!!"))
	}()

	// Builder
	func() {
		var b strings.Builder
		for i := 3; i > 0; i-- {
			fmt.Fprintf(&b, "%d...", i)
		}
		b.WriteString("ignition")
		fmt.Println(b.String())
	}()
}
