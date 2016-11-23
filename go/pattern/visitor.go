package main

import "fmt"

type CarPart interface{
	Accept(CarPartVisitor)
}

type Wheel struct{
	Name string
}

func (this *Wheel)Accept(visitor CarPartVisitor){
	visitor.visitWheel(this)
}

type Engine struct{}

func (this *Engine)Accept(visitor CarPartVisitor){
	visitor.visitEngine(this)
}

type Car struct{
	parts []CarPart
}

func NewCar()*Car{
	this := new(Car)
	this.parts = []CarPart{
		&Wheel{"form left"},
		&Wheel{"from right"},
		&Wheel{"rear right"},
		&Wheel{"rear left"},
		&Engine{},
	}
	return this
}

func (this *Car)Accept(visitor CarPartVisitor){
	for _,part := range this.parts {
		part.Accept(visitor)
	}
}

type CarPartVisitor interface {
	visitWheel(wheel *Wheel)
	visitEngine(engine *Engine)
}

type GetMessageVisitor struct{
	Messages []string
}

func (this *GetMessageVisitor)visitWheel(wheel *Wheel){
	this.Messages = append(this.Messages,fmt.Sprintf("Visiting the %v wheel.",wheel.Name))
}

func (this *GetMessageVisitor)visitEngine(engine *Engine){
	this.Messages = append(this.Messages,"Visiting engine.")
}

func main(){
	car := NewCar()
	visitor := new(GetMessageVisitor)
	car.Accept(visitor)
	fmt.Println(visitor.Messages)
}