#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/kprobes.h>

static struct kprobe kp = {
	.symbol_name = "kernel_clone",
};

static int pre_handler(struct kprobe *p, struct pt_regs *regs) {
	printk(KERN_INFO "pre_handler addr=%p\n", p->addr);
	return 0;
}

static void post_handler(struct kprobe *p, struct pt_regs *regs, unsigned long flags) {
	
}

static int fault_handler(struct kprobe *p, struct pt_regs *regs, int trapnr) {
	printk(KERN_INFO "fault_handler: p->addr = 0x%p, trap #%dn", p->addr, trapnr);
	return 0;
}

static int __init kprobe_init(void) {
	int ret;

	kp.pre_handler = pre_handler;
	kp.post_handler = post_handler;
	kp.fault_handler = fault_handler;

	ret = register_kprobe(&kp);
	if (ret < 0) {
		printk(KERN_WARNING "register_kprobe failed, returned %d\n", ret);
		return ret;
	}
	printk(KERN_INFO "Planted kprobe at %p\n", kp.addr);
	return 0;
}

static void __exit kprobe_exit(void) {
	unregister_kprobe(&kp);
	printk(KERN_INFO "kprobe at %p unregistered\n", kp.addr);
}

module_init(kprobe_init);
module_exit(kprobe_exit);
MODULE_LICENSE("GPL");
