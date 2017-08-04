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

t = {10,20,30}
for element in values(t) do
	print(element)
end


function allwords()
	local line = io.read()
	local pos = 1
	return	function()
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


t = {10,20,30,color="blue"}
for k,v in ipairs(t) do
	print(k,v)
end
for k,v in pairs(t) do
	print(k,v)
end


