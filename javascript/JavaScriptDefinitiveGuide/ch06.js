// 对象的创建可以通过对象直接量、关键字 new 和 Object.Create()函数来创建
// 对象直接量
var empty = {}
var point = {x:0,y:0}
// 通过new创建对象
var o = new Object()
var a = new Array()
var r = new RegExp("js")
// 通过Object.create()方法创建一个新对象,其中第一个参数是这个对象的原型
var o1 = Object.create({x:1,y:2}) // o1继承了属性x和y

// 
var o = {
    // 普通的数据属性
    data_prop:1,

    // 存取器属性都是成对定义的函数
    get accessor_prop(){},
    set accessor_prop(value){}
}


var p = {
    x:1.0,
    y:1.0,

    // r是可读写的存取器属性,它有getter和setter,
    // 函数体结束后不要忘记带上逗号
    get r(){
        return Math.sqrt(this.x * this.x + this.y * this.y)
    },

    set r(newvalue){
        var oldvalue = Math.sqrt(this.x * this.x + this.y * this.y)
        var ratio = newvalue / oldvalue
        this.x *= ratio
        this.y *= ratio
    },

    // theta是只读存取器属性,它只有getter方法
    get theta(){
        return Math.atan2(this.y,this.x)
    }
}

console.log(p.r)
console.log(p.theta)
p.r = 100
console.log(p.x,p.y)
p.theta = 3.14
console.log(p.theta)


var serialnum = {
    // 这个数据属性包含下一个序列号
    // $符号暗示这个属性是一个私有属性
    $n:0,

    get next(){
        return this.$n++
    },
    set next(n){
        if(n > this.$n){
            this.$n = n
        }
        else{
            throw "序列号的值不能比当前值小"
        }
        
    }
}

// 属性的特性
// 数据属性的4个特性分别是它的值(value),可写性(writable),可枚举型(enumerable)和可配置性(configurable)
// 存取器的4个特性是读取(get),写入(set),可枚举型和可配置性

console.log(Object.getOwnPropertyDescriptor({x:1},"x")) // {value:1,writable:true,enumerable:true,configurable:true}

var random = {
    get octet(){
        return Math.floor(Math.random() * 256)
    },
    get uint16(){
        return Math.floor(Math.random() * 65536)
    },
    get int16(){
        return Math.floor(Math.random() * 65536) - 32768
    }
}

console.log(Object.getOwnPropertyDescriptor(random,"octet"))

// 对于继承属性和不存在的属性,返回undefined
Object.getOwnPropertyDescriptor({},"x") // => undefined
Object.getOwnPropertyDescriptor({},"toString") // => undefined

// 设置属性的特性,需要调用Object.defineProperty()方法
var o = {} // 创建一个空对象
Object.defineProperty(o,"x",{value:1,
                             writebale:true,
                             enumerable:false,
                             configurable:true})
// 属性是存在的,但不可枚举
console.log("o.x = ",o.x) // => 1
console.log("Object.keys(o) = ",Object.keys(o))

// 现在对属性x做修改,让它变为只读
Object.defineProperty(o,"x",{writebale:false})

// 试图更改这个属性的值
o.x = 2; // 操作失败但不报错,而在严格模式中抛出类型错误异常
console.log("o.x = ",o.x)

