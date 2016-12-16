
// 复合语句和空语句

// 用花括号将多条语句括起来即形成一条复合语句
{
    x = Math.PI
    cx = Math.cos(x)
    console.log("cos(π) = " + cx)
}

// 空语句
;

// 声明语句
// var
var i = 0,greeting = "Hello";
// function
// 函数声明语句
function hypotenuse(x,y){
    return Math.sqrt(x * x  + y * y)
}

// 条件语句
// if
var username = "John"
if(username == null){
    username = "OkabeRintarou"
    console.log(username)
}
else{
    console.log("username = " , username)
}

// switch
function convert(x){
    switch(typeof x){
        case 'number':
            return x.toString(16)
        case 'string':
            return '"' + x + '"'
        default:
            return String(x)
    }
}

// 循环
// while
var count  = 0
while(count < 9){
    console.log(count)
    count++
}

// do/while
function printArray(a){
    var len = a.length,i = 0
    if(len == 0){
       console.log("Empty Array") 
    }
    else{
        do{
            console.log(a[i])
        }while(++i < len)
    }
}

var myArray = [0x11,0x22,0x33,0x44,0x55]
printArray(myArray)

// for
var i,j
var sum = 0
for(i = 0,j = 10;i < 10;i++,j--){
    sum += i * j
}

console.log("sum = ",sum)

// for/in
var o = {x:1,y:2,z:3}
var a = [],i = 0
for(a[i++] in o){
    ;
}

console.log(a)

// throw 语句
try{
}
catch(e){
}
finally{

}
