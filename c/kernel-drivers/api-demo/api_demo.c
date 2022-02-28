#include <linux/types.h>
#include <linux/list.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/sched.h>
#include "test_list.h"

static int kernel_api_demo_init(void) {
	test_list();
	return 0;
}

static void kernel_api_demo_exit(void) {
}

module_init(kernel_api_demo_init);
module_exit(kernel_api_demo_exit);
MODULE_LICENSE("GPL v2");
