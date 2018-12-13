
int Foo(int a,int b) {
	if (a == 0 || b == 0) {
		throw "don't do that";
	}

	int c = a % b;
	if (c == 0) {
		return b;
	}
	return Foo(b,c);
}


bool IsPrime(int n) {
	if (n <= 1) return false;

	if (n % 2 == 0) return false;

	for (int i = 3;;i += 2) {
		if (i > n / i) break;
		if (n % i == 0) return false;
	}

	return true;
}
