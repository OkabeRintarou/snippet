--[[
算数操作符
--]]

--	+　加
--	-　减或取负
--  *  乘
--  /  除
--  ^  冥

-- 实数的取模
x = math.pi
print(x % 1) -- math.pi的小数部分
print(x % 0.1) -- 0.04...
print(x - x % 1) -- math.pi的整数部分

--[[
关系操作符
--]]

-- < 小于
-- > 大于
-- <= 小于等于
-- >= 大于等于
-- == 相等性测试
-- ~= 不等性测试

print(nil == nil) -- nil 只与自身相等
print("hello" == 123) -- 类型不同不相等
print({} == {}) -- table,userdata和function作引用比较

--[[
逻辑操作符
--]]

-- and or not

--[[
字符串连接
--]]
print("Hello" .. "World")

--[[
table构造式
--]]
days = {}
days = {"Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"} -- days[1] = "Suday" -- 列表风格的初始化
a = {x=10,y=20} -- 等价于　a={};a.x=10;a.y=20　　记录风格的初始化
a.x=nil -- 删除字段

--[[ 构造链表
list = nil
for line in io.lines() do
	list = {next=list,value=line}
end
local l = list
while l do
	print(l.value)
	l = l.next
end
--]]

-- 混合记录风格和列表风格的初始化
polyline = {color="blue",thickness=2,npoints=4,
			{x=0,y=0},{x=-10,y=0},{x=-10,y=1},{x=0,y=1}}
print(polyline[2].x) -- -10

-- 通用的初始化
opnames = {["+"] = "add",["-"] = "sub",["*"] = "mul",["/"] = "div"}
i = 20;s = "-"
a = {[i+0] = s,[i+1] = s..s,[i+2] = s..s..s}
print(opnames[s]) --> sub
print(a[22]) --> ---


