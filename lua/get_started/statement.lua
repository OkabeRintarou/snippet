--[[
赋值
--]]
a = "hello".."world"
t = {n=1}
t.n = t.n + 1

-- 多重赋值
x,y = 10,20
print(x,y)

--[[
局部变量和块
--]]
x = 10
local i = 1

while i <= x do
	local x = i * 2
	print(x)
	i = i + 1
end

print('\n\n\n')
if i > 20 then
	local x
	x = 20
	print(x + 2) -- 局部变量
else
	print(x)     -- 全局变量
end

-- 通过do-end 显示界定一个块
do
	local a = 100
	print(a ^ 2)
end

--[[
控制结构
--]]

-- if
a = 10
if a < 0 then a = 0 
elseif a < 5 then a = 1
else a = 2
end
print('a = ',a)

-- repeat,for,while
a = 10
print('while:')
local i = 0
while i < a do
	print(i)
	i = i + 1
end

print('repeat:')
i = 0
repeat
	print(i)
	i = i + 1
until i >= 10

print('for:')
-- 数字型for
for i=1,10,-1 do
	print(i)
end
-- 泛型for
days = {"Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"} -- days[1] = "Suday" -- 列表风格的初始化
for i,v in ipairs(days) do
	print(i,v)
end

