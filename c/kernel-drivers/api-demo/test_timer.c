#include <linux/timer.h>
#include <linux/kernel.h>
#include <linux/jiffies.h>
#include <linux/delay.h>

static struct timer_list my_timer;

static void my_timer_callback(struct timer_list *) {
  pr_info("%s called (%ld).\n", __FUNCTION__, jiffies);
}

void test_timer_init(void) {
  int ret;

  timer_setup(&my_timer, &my_timer_callback, 0);
  ret = mod_timer(&my_timer, jiffies + msecs_to_jiffies(300));
  if (ret) {
    pr_warn("Timer filling failed\n");
  }
}

void test_timer_fini(void) {
  int ret;

  ret = del_timer(&my_timer);
  if (ret) {
    pr_warn("The timer is still in use...\n");
  } else {
    pr_info("Timer module unloaded\n");
  }
}