package main

import (
	"fmt"
	"unsafe"
	"sync/atomic"
	)

func main() {
	{
		// func AddInt32(addr *int32, delta int32) (new int32)
		{
			var num int32 = 100
			newNum := atomic.AddInt32(&num,99)
			fmt.Printf("num = %d,new = %d\n",num,newNum)
		}
		// func AddInt64(addr *int64, delta int64) (new int64)
		{
			var num int64 = 100
			newNum := atomic.AddInt64(&num,99)
			fmt.Printf("num = %d,new = %d\n",num,newNum)
		}

		// func AddUint32(addr *uint32, delta uint32) (new uint32)
		{
			var num uint32 = 100
			newNum := atomic.AddUint32(&num,^uint32(0))
			fmt.Printf("num = %d,new = %d\n",num,newNum)
		}

		// func AddUint64(addr *uint64, delta uint64) (new uint64)
		{
			var num uint64 = 100
			newNum := atomic.AddUint64(&num,^uint64(99-1)) // 100 - 99
			fmt.Printf("num = %d,new = %d\n",num,newNum)
		}

		// func AddUintptr(addr *uintptr, delta uintptr) (new uintptr)
		{
			var num uintptr = 100
			newNum := atomic.AddUintptr(&num,10)
			fmt.Printf("num = %d,new = %d\n",num,newNum)
		}

		// func CompareAndSwapInt32(addr *int32, old,new int32) (swapped bool)
		{
			var num int32 = 100
			if (atomic.CompareAndSwapInt32(&num,100,99)) {
				fmt.Println("Now num equal to 99")
			}
		}

		// func CompareAndSwapInt64(addr *int64, old,new int64) (swapped bool)
		// func CompareAndSwapPointer(addr *unsafe.Pointer, old,new unsafe.Pointer) (swapped bool)
		// func CompareAndSwapUint32(addr *uint32, old,new uint32) (swapped bool)
		// func CompareAndSwapUint64(addr *uint64, old,new uint64) (swapped bool)
		// func CompareAdnSwapUintptr(addr *uintptr, old,new uintptr) (swapped bool)

		// func LoadInt32(addr *int32) (val int32)
		{
			var num int32 = 100
			fmt.Println(atomic.LoadInt32(&num))
		}
		// func LoadInt64(addr *int64) (val int64)
		{
			var num int64 = 100
			fmt.Println(atomic.LoadInt64(&num))
		}
		// func LoadPointer(addr *unsafe.Pointer) (val unsafe.Pointer)
		{
			var num int = 100
			ptr := unsafe.Pointer(&num)
			fmt.Printf("%p\n",ptr)
		}
		// func LoadUint32(addr *uint32) (val uint32)
		// func LoadUint64(addr *uint64) (val uint64)
		// func LoadUintptr(addr *uintptr) (val uintptr)

		// func StoreInt32(addr *int32, val int32)
		{
			var num int32
			atomic.StoreInt32(&num,100)
			fmt.Println(num)
		}
		// func StoreInt64(addr *int64, val int64)
		// func StorePointer(addr *unsafe.Pointer, val unsafe.Pointer)
		// func StoreUint32(addr *uint32, val uint32)
		// func StoreUint64(addr *uint64, val uint64)
		// func StoreUintptr(addr *uintptr, val uintptr)

		// func SwapInt32(addr *int32, new int32) (old int32)
		{
			var num int32 = 100
			old := atomic.SwapInt32(&num,99)
			fmt.Printf("old = %d,new = %d\n",old,num)
		}

		// func SwapInt64(addr *int64, new int64) (old int64)
		// func SwapPointer(addr *unsafe.Pointer, new unsafe.Pointer) (old unsafe.Pointer)
		// func SwapUint32(addr *uint32, new uint32) (old uint32)
		// func SwapUint64(addr *uint64, new uint64) (old uint64)
		// func SwapUintptr(addr *uintptr, new uintptr) (old uintptr)

	}
}
