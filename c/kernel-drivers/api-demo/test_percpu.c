#include <linux/percpu.h>
#include <linux/printk.h>
#include "test_percpu.h"

struct birthday {
	int day;
	int month;
	int year;
};

static DEFINE_PER_CPU(struct birthday, my_birthday) = {7, 18, 1994};

void test_percpu(void) {
	get_cpu_var(my_birthday).year++;
	put_cpu_var(my_birthday);
}
