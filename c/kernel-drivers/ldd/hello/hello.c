#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/sched.h>


static int hello_init(void) {
	printk(KERN_ALERT "Hello, World!\n");
	printk(KERN_INFO "The process is \"%s\" (pid %i)\n", current->comm, current->pid);
	return 0;
}

static void hello_exit(void) {
	printk(KERN_ALERT "Goodbye!\n");
}

module_init(hello_init);
module_exit(hello_exit);
MODULE_LICENSE("GPL");
