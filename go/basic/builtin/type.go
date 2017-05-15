package main

func main(){
	// 运行时内存分配操作会确保变量自动初始化为零值
	var x int // 自动初始化为0
	println(x)
	var y = false // 自动推断为 bool 类型
	println(y)

	{
		var x,y int  // 相同类型的多个变量
		println(x,y) 
		var a,s = 100,"abc" // 不同类型初始化值
		println(a,s)
	}

	{
		// 以组方式整理多行变量定义
		var (
				x,y int
				a,s = 100,"abc"
			)
		println(x,y,a,s)
	}

	{
		// 简短模式
		// 限制: 1. 定义变量,同时显示初始化
		//       2. 不能提供数据类型
		//       3. 只能在函数内部使用
		x := 10
		println(&x)
		a,s := 1,"abc"
		println(x,a,s)

		x,y := 200,300 // x 退化为赋值操作,仅有 y 是变量定义
		// 退化赋值的条件是至少有一个新变量定义,且必须是同一作用域
		println(&x,y)
	}


	{
		const x = 100 // 未使用,不会引发编译错误
		{
			const(
					b byte = byte(x)
					n = uint(x)
				)
		}
	}


	{
		// 枚举
		const (
				x = iota // 0
				y		 // 1
				z		 // 2
			)

		{
			const (
					_ = iota
					KB = 1 << (10 * iota) // 1 << (10 * 1)
					MB  // 1 << (10 * 2)
					GB  // 1 << (10 * 3)
				  )
		}

		{
			const(
					_,_ = iota,iota * 10	// 0,0 * 10
					a,b						// 1,1 * 10
					c,d						// 2,2 * 10
				)
		}

		{
			// 实际编码中,建议用自定义类型实现用途明确的枚举类型
			type color byte
			const (
					black color = iota
					red
					blue
				)
			println(black,red,blue)
		}
	}


	// 引用类型,特指 slice,map,channel 三种预定义类型

}
