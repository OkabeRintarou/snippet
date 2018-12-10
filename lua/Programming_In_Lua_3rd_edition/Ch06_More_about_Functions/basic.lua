network = {
	{name="grauna",IP="210.26.30.34"},
	{name="arraial",IP="210.26.30.23"},
	{name="lua",IP="210.26.23.12"},
	{name="derain",IP="210.26.23.20"},
}

local function print_table(t)
	io.write('{')
	for k,v in pairs(t) do
		if type(v) == 'table' then
			print_table(v)
		else
			io.write(v)
		end
		io.write(' ')
	end
	io.write('}\n')
end

print_table(network)
table.sort(network,function(a,b) return (a.name > b.name) end)
print_table(network)

function derivative(f, delta)
	delta = delta or 1e-4
	return 
		function(x)
			return (f(x+delta) - f(x)) / delta
		end
end

c = derivative(math.sin,1e-12)
print(math.cos(5.2),c(5.2))

