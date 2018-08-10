package main

import (
	"fmt"
	"math/rand"
)

const (
	MsgConnect = iota
	MsgData
)

type Message struct {
	Type    int
	Size    int
	Payload []byte
}

func NewConnect() *Message {
	return &Message{Type: MsgConnect}
}

func NewData(payload []byte) *Message {
	return &Message{Type: MsgData, Size: len(payload), Payload: payload}
}

func (m *Message) String() string {
	var name, payload string

	switch m.Type {
	case MsgConnect:
		name = "Connect"
	case MsgData:
		name = "Data"
		payload = " " + string(m.Payload)
	}
	return fmt.Sprintf("[%s %s]", name, payload)
}

func randomBytes(length int) []byte {
	if length <= 0 {
		return nil
	}
	b := make([]byte, length)
	var c byte
	for i := 0; i < length; i++ {
		if rand.Intn(2) == 0 {
			c = byte('a' + rand.Intn(26))
		} else {
			c = byte('A' + rand.Intn(26))
		}
		b[i] = c
	}
	return b
}
