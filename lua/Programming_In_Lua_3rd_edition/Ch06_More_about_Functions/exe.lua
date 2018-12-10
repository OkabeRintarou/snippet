function integral(f,a,b,delta)
	delta = delta or 1e-4
	local area = 0
	for i = a,b,delta do
		area = area + f(i) * delta
	end
	return area
end

print(integral(function (x) return x end, 0,10))
print(integral(function (x) return x * x / 2 end, 0,10))
local function f(x) return 1.0 /6 * x * x * x end
print(f(10))

function newpoly(t)
	return 
	function(x)
		local s,n = t[1],#t
		local mul = x
		for i = 1,n - 1 do
			s = s + mul * t[i + 1]
			mul = mul * x
		end
		return s
	end
end

f = newpoly({3,0,1})
print(f(0))
print(f(5))
print(f(10))


n = math.random(123456789)
function f()
	n = n - 1
	if n < 0 then
		return nil
	else
		return 'i=1;'
	end
end

load(f)()
