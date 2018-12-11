function values(t)
	local i = 0
	return function() i = i + 1;return t[i] end
end

t = {10,20,30}
iter = values(t)
while true do
	local element = iter()
	if element == nil then break end
	print(element)
end

for element in values(t) do
	print(element)
end

function allwords()
	local line = io.read()
	local pos = 1
	return function()
		while line do
			local s,e = string.find(line,"%w+",pos)
			if s then 
				pos = e + 1
				return string.sub(line,s,e)
			else
				line = io.read()
				pos = 1
			end
		end
		return nil
	end
end

function test_allwords()
	for word in allwords() do
		print(word)
	end
end

function getnext(list, node)
	if node == nil then
		return list
	else
		return node.next
	end
end

function traverse(list)
	return getnext, list, nil
end

function test_traverse()
	list = nil
	for line in io.lines() do
		list = {val = line,next = list}
	end

	for node in traverse(list) do
		print(node.val)
	end
end

local iterator

function allwords2()
	fp = io.input('basic.lua')
	local state = {file = fp,line = fp:read(), pos = 1}
	return iterator, state
end

function iterator(state)
	while state.line do
		-- search for next word
		local s,e = string.find(state.line, "%w+", state.pos)
		if s then 
			-- update next position
			state.pos = e + 1
			return string.sub(state.line,s,e)
		else
			state.line = state.file:read()
			state.pos = 1
		end
	end
	return nil
end

function test_allwords2()
	for word in allwords2() do
		print(word)
	end
end

--test_allwords()
--test_traverse()
--test_allwords2()

