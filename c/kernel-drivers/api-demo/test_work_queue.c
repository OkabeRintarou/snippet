#include <linux/workqueue.h>
#include <linux/wait.h>
#include <linux/delay.h>
#include <linux/slab.h>
#include "test_work_queue.h"

static int sleep = 0;

struct work_data {
  struct work_struct my_work;
  wait_queue_head_t my_wq;
  int the_data;
};

static void work_handler(struct work_struct *work) {
  struct work_data *my_data = container_of(work, struct work_data, my_work);
  pr_info("Work queue module handler: %s, data is %d\n", __FUNCTION__ , my_data->the_data);
  msleep(2000);
  sleep = 1;
  wake_up_interruptible(&my_data->my_wq);
}

static void test_system_wq(void) {
  struct work_data *my_data;

  my_data = kmalloc(sizeof(struct work_data), GFP_KERNEL);
  my_data->the_data = 34;

  INIT_WORK(&my_data->my_work, work_handler);
  init_waitqueue_head(&my_data->my_wq);

  schedule_work(&my_data->my_work);
  pr_info("I'm going to sleep ...\n");
  wait_event_interruptible(my_data->my_wq, sleep != 0);
  pr_info("I am Waked up...\n");
  kfree(my_data);
}

struct workqueue_struct *wq;

static void custom_work_handler(struct work_struct *work) {
  struct work_data *my_data = container_of(work, struct work_data, my_work);
  pr_info("Work queue module handle: %s, data is %d\n", __FUNCTION__ , my_data->the_data);
}

static void test_custom_wq(void) {
  struct work_data *my_data;

  pr_info("Work queue module init: %s %d\n", __FUNCTION__, __LINE__);
  wq = create_singlethread_workqueue("my_single_thread");

  my_data = kmalloc(sizeof(struct work_data), GFP_KERNEL);
  my_data->the_data = 100;
  INIT_WORK(&my_data->my_work, custom_work_handler);
  queue_work(wq, &my_data->my_work);
}

void test_work_queue_init(void) {
  test_system_wq();
  test_custom_wq();
}

void test_work_queue_fini(void) {
  if (wq) {
    flush_workqueue(wq);
  }
}