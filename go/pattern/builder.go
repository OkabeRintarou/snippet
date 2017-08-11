package main

import (
	"fmt"
	"errors"
)

var errIncompleteCar = errors.New("incomplete car")

type Speed float64

const (
	MPH Speed	= 1
	KPH			= 1.60934
)

type Color string

const (
	BlueColor		Color = "blue"
	GreenColor		Color = "green"
	RedColor		Color = "red"
)

type Wheels string

const (
	SportsWheels		Wheels = "sports"
	SteelWheels			Wheels = "steel"
)

type Builder interface {
	Paint(Color) Builder
	Wheels(Wheels) Builder
	TopSpeed(Speed) Builder
	Build() Interface
}

type Interface interface {
	Drive() error
	Stop() error
}


type CarBuilder struct {
	color		Color
	wheels		Wheels
	topSpeed	Speed
}

func NewBuilder() *CarBuilder {
	return &CarBuilder{}
}

func (cb *CarBuilder)Paint(color Color) *CarBuilder {
	cb.color = color
	return cb
}

func (cb *CarBuilder)Wheels(wheels Wheels) *CarBuilder {
	cb.wheels = wheels
	return cb
}

func (cb *CarBuilder)TopSpeed(speed Speed) *CarBuilder {
	cb.topSpeed = speed
	return cb
}

func (cb *CarBuilder)Build() Interface {
	return cb
}

func (cb *CarBuilder)Drive() error {
	if cb.wheels == "" || cb.topSpeed == 0 || cb.color == "" {
		return errIncompleteCar
	}
	fmt.Printf("Car with %s wheels %s color drive at top speed %f\n",cb.wheels,cb.color,cb.topSpeed)
	return nil
}

func (cb *CarBuilder)Stop() error {
	fmt.Println("Car stop")
	return nil
}


func main() {
	assembly := NewBuilder().Paint(RedColor)

	familyCar := assembly.Wheels(SportsWheels).TopSpeed(50 * MPH).Build()
	familyCar.Drive()
	sportsCar := assembly.Wheels(SteelWheels).TopSpeed(150 * MPH).Build()
	sportsCar.Drive()
}
