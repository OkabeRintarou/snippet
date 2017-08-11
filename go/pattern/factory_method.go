package main

import (
	"io"
)

type Store interface {
	Open(string) (io.ReadWriteCloser,error)
}

type StorageType int

const (
	DiskStorage StorageType = 1 << iota
	TempStorage
	MemoryStorage
)

func NewStore(t StorageType) Store {
	switch(t) {
	case MemoryStorage:
		return newMemoryStorage()
	case DiskStorage:
		return newDiskStorage()
	case TempStorage:
		return TempStorage()
	}
}

func main() {
	s,_ := NewStorage(DiskStorage)
	f,_ := s.Open("file")

	n,_ := f.Write([]byte("data"))
	defer f.Close()
}
