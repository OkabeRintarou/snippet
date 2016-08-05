var x; // 变量通过var关键字声明
x = 0; // 值通过等号赋值给变量

// Javascript支持多种数据类型
x = 1;      // 数字
x = 0.01;   // 整数和实数公用一种数据类型
x = "hello world"; // 由双引号内的文本构成的字符串
x = 'Javascript';  // 单引号内的文本同样构成字符串
x = true;           // 布尔值
x = null;           // null 是一个特殊的值
x = undefined;      // undefined 和 null 非常类似

// Javascript中的最重要的类型就是对象
// 对象是名/值对的集合,或字符串到值映射的集合
var book = { // 对象由花括号括起来的
    topic: "Javascript",
    fat: true
};

// 通过 "." 或 "[]"来访问对象属性
book.topic
book["fat"]
book.author = "Flanagan"; // 通过赋值创建一个新属性
book.contents = {};       // {} 是一个空对象,它没有属性


// Javascript同样支持数组(以数字为索引的列表)
var primes = [2, 3, 5, 7];
primes[0]
primes.length
primes[primes.length - 1] // =>7:数组中的最后一个元素
primes[4] = 9;              // 通过赋值来添加元素
var empty = []//[] 是空数组,它具有0个元素
empty.length  // => 0

var points = [
    { x: 0, y: 0 },
    { x: 1, y: 1 }
];
var data = {
    trial1: [[1, 2], [3, 4]],
    trial2: [[2, 3], [4, 5]]
};


// Javascript 运算符
3 + 2
3 - 2
3 * 2
3 / 2
points[1].x - points[0].x
"3" + "2"

// 一些算术运算符的简写形式
var count = 0;
count++;
count--;
count += 2;
count *= 3;
count

var x = 2, y = 3;
x == y
x != y
x < y
x <= y
x > y
x >= y
"two" == "three"
"two" > "three"
false == (x > y)

// 逻辑运算符
(x == 2) && (y == 3)
(x > 3) || (y < 3)
!(x == y)

// 函数
function plus1(x){
    return x + 1
}
plus1(x)

var square = function (x){
    return x * x;
}

// 当函数赋值给对象的属性,我们称为"方法"
var a = []
a.push(1, 2, 3)
a.reverse();

// 我们也可以定义自己的方法,"this"关键字是对定义方法的对象的引用
points.dist = function (){
    var p1 = this[0]
    var p2 = this[1]
    var a = p2.x - p1.x
    var b = p2.y - p1.y
    return Math.sqrt(a * a  + b * b)
}
points.dist()

// 条件判断语句
function abs(x){
    if (x >= 0) {
        return x;
    } else {
        return -x;
    }
}