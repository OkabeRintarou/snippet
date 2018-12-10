function concat(...)
	local s = ""
	for k,v in ipairs{...} do
		s = s..v
	end
	return s
end

print(concat("a","b"))
print(concat("a","b","c"))
print(concat("a","b","c","d"))

function exe5_2(array)
	print(table.unpack(array))
	for k,v in ipairs(array) do
		print(v)
	end
end

exe5_2({100,200,300})

function exe5_3(first,...)
	return ...
end
print(exe5_3(100,200))
print(exe5_3(100))

function print_table(tab)
	io.write('{ ')
	for k,v in pairs(tab) do
		if type(v) == 'table' then
			print_table(v)
			io.write(' ')
		else
			io.write(v .. ' ')
		end
	end
	io.write('}')
end

function deep_copy(tab)
	r = {}
	index = 1
	for k,v in pairs(tab) do
		if type(v) == 'table' then 
			r[index] = deep_copy(v)
		else
			r[index] = v
		end
		index = index + 1
	end
	return r
end

function comb(tab1,tab2)

	if #tab1 == 0 or #tab2 == 0 then
		return tab2
	end
	local r = deep_copy(tab1)
	for k,v in pairs(tab2) do
		r[#r+1] = v
	end
	return r
end


function combinations(array, m)
	local n = #array
	if n < m then
		return nil
	elseif m == 0 then
		return {}
	elseif m == n then
		return {deep_copy(array)}
	else
		local first = array[1]
		local r = {}

		array[1] = nil
		local rest = combinations(deep_copy(array), m - 1)

		if rest ~= nil and #rest > 0 then
			for k,v in pairs(rest) do
				r[#r+1] = comb({first},v)
			end
		else
			r[#r+1] = {first}
		end

		rest = combinations(deep_copy(array), m)
		if rest ~= nil then
			for k,v in pairs(rest) do
				r[#r + 1] = v
			end
		end
		return r		
	end
end


print_table(combinations({"a","b","c"},2))
print()
print_table(combinations({"a","b","c","d"},3))
print()
print_table(combinations({"a","b","c","d"},2))
