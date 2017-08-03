--[[
function foo(x) return 2 * x end 等价于
--]]
foo = function(x) return 2 * x end
-- 一个函数定义实际上就是一条赋值语句
print(foo(100))


network = {
	{name="grauna",IP="210.26.30.34"},
	{name="arraial",IP="210.26.30.23"},
	{name="lua",IP="210.26.23.12"},
	{name="derain",IP="210.26.23.20"},
}
table.sort(network,function (a,b) return (a.name < b.name) end)
for k,v in pairs(network) do
	print(v.name,v.IP)
end

--[[
local fact = function(n)
	if n == 0 then return 1
	else return n * fact(n - 1) 此处局部fact尚未定义完毕,此处表达式调用了一个全局的fact
	end
end
fact(5)
--]]
-- 正确做法
local fact
fact = function(n)
	if n == 0 then return 1
	else return n * fact(n - 1)
	end
end
print(fact(5))

-- 尾递归
function g(x) end
function f(x) return g(x) end
-- 下面的代码中不是尾递归
function f1(x) g(x) end -- f1不能立即返回,它需要丢弃g返回的临时结果
function f2(x) return g(x) + 1 end -- 必须做一次加法
function f3(x) return x or g(x) end
function f4(x) return (g(x)) end 
