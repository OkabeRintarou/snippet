// Javascript 的数据类型分为:原始类型和对象类型

// 原始类型包括数字、字符串和布尔值

// 数字
var integer = 0xff
var float = 1.47e-32

// Javascript支持 + - * / 和 % 运算符
Number.POSITIVE_INFINITY
Number.NEGATIVE_INFINITY 
console.log(1 / 0) 

console.log(isFinite(1 / 0))   // false
console.log(isNaN(0 / 0)) // true

// 浮点精度
var x = .3 - .2
var y = .2 - .1
console.log( x == y ) // false

// 日期和时间
var then = new Date(2016, 8, 5)  // 2016年8月5日
var later = new Date(2016, 8, 5, 20, 30, 0)// 2016年8月5日8:30:00pm
var now = new Date()// 当前日期
var elapsed = now - then// 日期减法，计算时间间隔的毫秒数
later.getFullYear()  // => 2016
later.getMonth()     // => 7:从0开始计数的月份
later.getDate()      // 从1开始计数的天数 
later.getDay()       // 得到星期几,0代表星期日,5代表星期一
later.getHours()     // 当地时间
later.getUTCHours()  // 使用UTC表示小时的时间，基于时区

// 文本
msg = "Hello, " + "world"
console.log(msg.length)
msg.charAt(0)    // => "H":第一个字符
msg.charAt(msg.length - 1) // => "d":最后一个字符
msg.substring(1, 4)      // => "ell":第2~4个字符
msg.slice(1, 4)            // => "ell"：同上
msg.slice(-3)              // => "rld":最后三个字符
msg.indexOf("l")           // => 2:字符l首次出现的位置
msg.lastIndexOf("l")       // => 10:字符l最后一次出现的位置
msg.indexOf("l", 3)         // => 3:在位置3及之后首次出现字符l的位置
msg.split(", ")              // => ["Hello","world"]分割成子串
msg.replace("H", "h")          // => hello, world
msg.toUpperCase()              // => "HELLO, WORLD"

var s = "test"
s.len = 4
var t = s.len
console.log(t == undefined) 
// 类型转换
10 + " objects"     // => "10 objects".数字10转换成字符串
"7" * "4"           // => 28L两个字符串均转换为数字
var n = 1 - "x"     // => NaN:字符串"x"无法转换为数字
n + " objects"      // => "NaN objects":NaN转换为字符串"NaN"

// 转换和相等性
console.log("null ==  undefined => ",null == undefined) // => true,这两值被认为相等
console.log("\"0\" == 0 => ","0" == 0)  // 在比较之前字符串转换为数字
console.log("0 == false => ",0 == false) // 在比较之前布尔值转换为数字
console.log("\"0\" == false => ","0" == false) // 在比较之前字符串和布尔值都转换为数字

// 显示类型转换
// 当不使用new运算符调用Boolean()、Number()、String()或Object()这些函数时，它们会作为类型转换函数做类型转换
Number("3")                 // => 3
String(false)               // -> "false" 
Boolean([])                 // => true
Object(3)                   // => new Number(3)


//
console.log({x:1,y:2}.toString())       // => "[object Object]"
[1,2,3].toString()                      // => "[1,2,3]"
(function(x){f(x);}).toString()         // => "function(x){f(x);}
/\d+/g.toString()                       // => "/\\d+/g"

