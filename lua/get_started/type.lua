--[[
8种基本类型
1. nil
2. boolean
3. number
4. string
5. userdata
6. function
7. thread
8. table
--]]

print(type(nil))				-- nil
print(type(true))				-- boolean
print(type(3.14*10))			-- number
print(type("HelloWorld")) 		-- string
print(type(print))				-- function

print(type({}))

print('\n\n')
--[[
boolean:false和nil为假,除此以外都为真
--]]

if not nil then
	print("<nil> is false")
end
if not false then
	print("<false> is false")
end
if "" then
	print("<\"\"> is true")
end
--[[
字符串
--]]
a = "one thing"
b = string.gsub(a,"one","another")
print(a,b)

a = [=[
<html>
<head><title>Hello</title></head>
<body><h1>World</h1></body>
</html>
a=b[c[i]]
]=]
print(a)

print(10 .. 20)
print(#"hello") -- #获得字符串的长度

--[[
table:关联数组
--]]
a = {}
k = "x"
a[k] = 10
a[20] = "target"
print(a["x"])
a[k] = a[k] + 1
print(a[k])
print(a.x)
a = {}
for i = 1,10 do
	a[i] = i * 100
end
for i = 1,#a do -- # 获得表的大小
	print("a[",i,"] = ",a[i])
end
print(a[#a]) -- 打印表的最后一个值
a[#a] = nil	 -- 删除最后一个值
a[#a + 1] = 1234 -- 在表的末尾添加值
print(a["yyy"])

a = {}
a[100000] = 1
print(#a) -- 打印0,长度操作符以nil作为终结符

--[[
函数
--]]

function fib(n)
	if n < 0 then return nil end
	if n == 1 or n == 2 then
		return 1
	end
	return fib(n - 1) + fib(n - 2)
end
print(fib(10))

