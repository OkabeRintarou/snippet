function fromto(s,e)
	local i = s
	return function()
		r = i
		i = i + 1
		if r > e then
			return nil
		end
		return r
	end
end

for i in fromto(10,12) do
	print(i)
end

function fromto_stateless(s,e)
	local function iter(e,current)
		if current + 1 > e then
			return nil
		else
			return current + 1
		end
	end
	return iter,e,s - 1
end

for i in fromto_stateless(20,25) do
	print(i)
end

function fromto_step(s,e,step)
	return function(state, current)
		if current + step > state.e then
			return nil
		else
			return current + step
		end
	end, {e = e, step = step},s - step
end

print('step:')
for i in fromto_step(20,30,2) do
	print(i)
end

function allwords()
	fp = io.input('basic.lua')
	local function iter(state)
		while state.line do
			local s,e = string.find(state.line, "%w+", state.pos)
			if s then
				state.pos = e + 1
				local sub = string.sub(state.line,s,e)
				if state.words[sub] == nil then
					state.words[sub] = sub
					return sub
				end
			else
				state.line = state.file:read()
				state.pos = 1
			end
		end
	end

	return iter, {file=fp,line = fp:read(), pos = 1, words = {}}
end

for word in allwords() do
	print(word)
end

function noempty_substring(str)
	local function iter(state)
		if state.e > state.len then
			state.s = state.s + 1
			if state.s > state.len then
				return nil
			end
			state.e = state.s
		end
		sub = string.sub(state.str,state.s,state.e)
		state.e = state.e + 1
		return sub
	end
	return iter, {str=str,len=#str,s=1,e=1}
end

for str in noempty_substring("abcde") do
	print(str)
end
