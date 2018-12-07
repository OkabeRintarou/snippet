function polynomial(p,x)
	n = #p
	s = p[1]
	for i = 1,n-1 do
		s = s + p[i+1] * (x ^ i)
	end
	return s
end

function polynomial1(p,x)
	s = p[1]
	n = #p
	mul = x

	for i = 1,n-1 do
		s = s + mul * p[i+1]	
		mul = mul * x
	end
	return s
end

print(polynomial({1,2,3,4},2))
print(polynomial1({1,2,3,4},2))

function is_boolean(v)
	return v == true or v == false
end

do
x = 1
print(is_boolean(x))
x = true
print(is_boolean(x))
x = false
print(is_boolean(x))
print(is_boolean("hello"))
end

sunday = "monday";monday = "sunday"
t = {sunday="monday",[sunday]=monday}  -- {"sunday":"monday","monday":"sunday"}
print(t.sunday,t[sunday],t[t.sunday])  --> monday sunday sunday
escapes = {["\a"] = "bell",["\b"] = "back space",
					 ["\f"] = "form feed",["\n"] = "newline",
					 ["\r"] = "carriage return",["\t"] = "horizontal tab",
					 ["\v"] = "vertical tab",["\\"] = "blackslash",
					 ["\""] = "double quote",["\'"] = "single quote"}

print(escapes["\'"])
