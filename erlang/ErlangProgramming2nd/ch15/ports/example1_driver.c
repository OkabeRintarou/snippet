#include <stdio.h>
#include <stdlib.h>

typedef unsigned char byte;

int read_cmd(byte *buff);
int write_cmd(byte *buff, int len);
int sum(int x, int y);
int twice(int x);

int main() {
	int fn, arg1, arg2, result;
	byte buf[100];

	while (read_cmd(buf) > 0) {
		fn = buf[0];

		if (fn == 1) {
			arg1 = buf[1];
			arg2 = buf[2];

			fprintf(stderr, "calling sum %i %i\n", arg1, arg2);
			result = sum(arg1, arg2);
		} else if (fn == 2) {
			arg1 = buf[1];
			result = twice(arg1);
		} else {
			exit(EXIT_FAILURE);
		}

		buf[0] = result;
		write_cmd(buf, 1);
	}

	return 0;
}
