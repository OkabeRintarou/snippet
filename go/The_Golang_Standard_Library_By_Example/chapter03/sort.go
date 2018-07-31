package main

import (
	"fmt"
	"sort"
)

type Person struct {
	Name string
	Age  int
}

type Class []Person

func (c Class) Len() int {
	return len(c)
}

func (c Class) Less(i, j int) bool {
	return c[i].Name < c[j].Name
}

func (c Class) Swap(i, j int) {
	c[i], c[j] = c[j], c[i]
}

func GuessingGame() {
	var s string
	fmt.Printf("Pick an integer from 0 to 100.\n")
	answer := sort.Search(100, func(i int) bool {
		fmt.Printf("Is your number <= %d\n", i)
		fmt.Scanf("%s", &s)
		return s != "" && s[0] == 'y'
	})
	fmt.Printf("Your number is %d.\n", answer)
}

func main() {
	c := make(Class, 0, 4)
	c = append(c, Person{"syl", 26})
	c = append(c, Person{"mayun", 40})
	c = append(c, Person{"ali", 22})
	fmt.Println("Default:\n\t", c)
	sort.Sort(c)
	fmt.Printf("IsSorted? %t\n", sort.IsSorted(c))
	fmt.Println("Sorted:\n\t", c)

	x := 11
	s := []int{3, 6, 8, 11, 45}
	pos := sort.Search(len(s), func(i int) bool { return s[i] >= x })
	if pos < len(s) && s[pos] == x {
		fmt.Printf("s[%d] = %d\n", pos, x)
	} else {
		fmt.Printf("%d not in s\n", x)
	}

	// GuessingGame()

	func() {
		s := []int{5, 2, 6, 3, 1, 4}
		sort.Ints(s)
		fmt.Println(s)

		s = []int{5, 2, 6, 3, 1, 4}
		sort.Sort(sort.Reverse(sort.IntSlice(s)))
		fmt.Println(s)
	}()
}
