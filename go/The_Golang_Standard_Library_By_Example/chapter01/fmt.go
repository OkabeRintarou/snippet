package main

import (
	"bytes"
	"fmt"
	"strconv"
)

type Person struct {
	Name string
	Age  int
	Sex  int
}

func (p *Person) String() string {
	buf := bytes.NewBufferString("This is ")
	buf.WriteString(p.Name + ", ")
	if p.Sex == 0 {
		buf.WriteString("He ")
	} else {
		buf.WriteString("She ")
	}
	buf.WriteString("is ")
	buf.WriteString(strconv.Itoa(p.Age))
	buf.WriteString(" years old.")
	return buf.String()
}

/*
func (p *Person) Format(f fmt.State, c rune) {
	if c == 'L' {
		f.Write([]byte(p.String()))
		f.Write([]byte(" Person has three fields."))
	} else {
		f.Write([]byte(fmt.Sprintln(p.String())))
	}
}
*/

func (p *Person) GoString() string {
	return "&Person{Name is " + p.Name + ", Age is " + strconv.Itoa(p.Age) + ", Sex is " + strconv.Itoa(p.Sex) + "}"
}

type Car struct {
	year int
	make string
}

func (c *Car) String() string {
	return fmt.Sprintf("{make:%s, year:%d}", c.make, c.year)
}

func main() {
	p := &Person{"syl", 26, 0}
	fmt.Printf("%#v\n", p)
	myCar := Car{year: 1996, make: "Toyota"}
	fmt.Println(&myCar)
}
