print(os.date())

s,e = string.find("hello Lua users","Lua")
print(s,e)


function maximum(a)
	local mi = 1
	local m = a[mi]
	for i,val in ipairs(a) do
		if val > m then
			mi = i;m = val
		end
	end
	return m,mi
end

max,index = maximum{100,1000,8,8,7,5}
print(max,index)

function foo0() end
function foo1() return "a" end
function foo2() return "a","b" end

print(foo2())
print(foo2(),1)
print(1,foo2())

-- (foo2()) 放在圆括号中迫使它只返回一个结果
print((foo2()))

function sum(...)
	local s = 0
	for i,v in ipairs{...} do
		s = s + v
	end
	return s
end

print(sum(1,2,3,4,5,6))

-- select
function foo(...)
	for i = 1,select('#',...) do
		local arg = select(i,...)
		print(i,'->',arg)
	end
end
foo(nil,100,nil,200)

-- 具名实参
function rename(arg)
	return os.rename(arg.old,arg.new)
end
rename{old='/tmp/old.lua',new='/tmp/new.lua'}

function Window(options)
	if type(options.title) ~= 'string' then
		error('no title')
	elseif type(options.width) ~= 'number' then
		error('no width')
	elseif type(options.height) ~= 'number' then
		error('no height')
	end
end

