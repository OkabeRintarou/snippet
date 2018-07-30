package main

import (
	"bytes"
	"fmt"
)

func main() {
	var b bytes.Buffer
	for i := 3; i > 0; i-- {
		fmt.Fprintf(&b, "%d...", i)
	}
	b.WriteString("ignition")
	fmt.Println(b.String())
}
