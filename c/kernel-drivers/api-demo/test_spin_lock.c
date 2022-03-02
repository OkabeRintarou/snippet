#include <linux/spinlock.h>
#include <linux/kthread.h>
#include <linux/sched.h>
#include "test_spin_lock.h"

struct task_struct *g_task0, *g_task1;
int g_count = 0;
int g_done = 0;
spinlock_t g_spin_lock;
spinlock_t g_done_lock;
struct task_struct *main_task;

int kernel_thread(void *) {
	int i, j;

	printk(KERN_INFO "%s[%llu] start run\n", current->comm, (unsigned long long)(current));

	for (i = 0; i < 1000; i++) {
		spin_lock(&g_spin_lock);
		for (j = 0; j < 200; j++) {
			++g_count;
		}
		spin_unlock(&g_spin_lock);
	}

	spin_lock(&g_done_lock);
	++g_done;
	spin_unlock(&g_done_lock);	

	wake_up_process(main_task);
	return 0;
}

void test_spin_lock_init(void) {
	int done, final_count;

	main_task = current;
	printk(KERN_INFO "main_task: %llu\n", (unsigned long long)(main_task));

	spin_lock_init(&g_spin_lock);

	g_task0 =  kthread_create(kernel_thread, NULL, "my-task0");
	if (!IS_ERR(g_task0))
		wake_up_process(g_task0);

	g_task1 =  kthread_create(kernel_thread, NULL, "my-task1");
	if (!IS_ERR(g_task1))
		wake_up_process(g_task1);

	for (;;) {

		spin_lock(&g_done_lock);
		done = g_done;
		spin_unlock(&g_done_lock);

		if (done == 2)
			break;

		set_current_state(TASK_INTERRUPTIBLE);
		schedule();
	}

	spin_lock(&g_spin_lock);
	final_count = g_count;
	spin_unlock(&g_spin_lock);
	printk(KERN_INFO "final value: %d\n", final_count);
}

void test_spin_lock_fini(void) {
}
