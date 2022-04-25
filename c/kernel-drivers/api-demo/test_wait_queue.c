#include <linux/printk.h>
#include <linux/wait.h>
#include <linux/sched.h>
#include <linux/time.h>
#include <linux/delay.h>
#include <linux/workqueue.h>
#include "test_wait_queue.h"

static DECLARE_WAIT_QUEUE_HEAD(my_wq);
static int condition = 0;

/* declare a work queue */
static struct work_struct wrk;

static void work_handler(struct work_struct*wrk) {
  pr_info("Waitqueue module handler %s\n", __FUNCTION__);
  msleep(5000);
  pr_info("Wake up the sleeping module\n");
  condition = 1;
  wake_up_interruptible(&my_wq);
}

void test_wait_queue(void) {
  pr_info("Wait queue example\n");

  INIT_WORK(&wrk, work_handler);
  schedule_work(&wrk);

  pr_info("Going to sleep %s\n", __FUNCTION__);
  wait_event_interruptible(my_wq, condition != 0);

  pr_info("woken up by the work job\n");
}
