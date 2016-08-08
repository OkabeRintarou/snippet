// Javascript 原始表达式包含常量或直接量、关键字和变量
// 直接量
1.23  // 数字直接量
"hello" // 字符串直接量

// Javascript中的一些保留字构成了原始表达式
true // 布尔值,真
false // 布尔值,假
null // 返回一个值:空
this // 返回当前对象

// 变量
var i = 100
i  // 返回变量i的值
undefined; // undefined是全局变量,和null不同,它不是一个关键字


// 对象和数组的初始化表达式
[]  // 一个空数组
[1 + 2,3 + 4] // 拥有两个元素的数组,第一个是3,第二个是7
var sparseArray = [1,,,,5]; //数组直接量中的列表逗号之间的元素可以省略,这时省略的空位会填充值undefined

var p = {x:2.3,y:-1.2} // 一个拥有两个属性成员的对象

// 函数定义表达式
var square =  function(x){return x * x}

// 属性访问表达式
p.x
p["x"]

// 调用表达式
console.log(Math.max(1,2,3))

// 对象创建表达式
new Object()
new Object  // 如果对象创建表达式不需要传入任何参数给构造函数的话,那么这对空圆括号可以省略

console.log("1" == true) // => true,首先将true转换为1,再将"1"转换为1

// in 运算符
var point = {x:1,y:2}
console.log("x" in point) // in运算符左操作数是一个字符串或可以转换为字符串,它的右操作数是对象
var data = [1,2,3]
"0" in data // => true
1 in data // => true
3 in data // => false

// instanceof 运算符
var a = [1,2,3]
a instanceof Array // => true,a是一个数组
a instanceof Object  // => true,所有的数组都是对象
a instanceof RegExp  // => false,数组不是正则表达式

// 表达式计算
/*
var geval = eval
var x = "global",y = "global";
function f(){
    var x = "local"
    eval("x += 'changed';")
    return x
}

function g(){
    var y = "local"
    geval("y += 'changed';")
    return y
}

console.log(f(),x)
console.log(g(),y)
*/

// typeof 运算符

// delete 运算符
var o = {x:1,y:2}
delete o.x
console.log("x" in o) // => false
