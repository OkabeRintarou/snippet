#include <linux/types.h>
#include <linux/list.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/sched.h>
#include "test_list.h"
#include "test_spin_lock.h"
#include "test_dma_fence.h"
#include "test_kmem_cache.h"
#include "test_percpu.h"
#include "test_container_of.h"
#include "test_wait_queue.h"
#include "test_timer.h"

static int kernel_api_demo_init(void) {
  test_timer_init();
  return 0;
}

static void kernel_api_demo_exit(void) {
  test_timer_fini();
}

module_init(kernel_api_demo_init);
module_exit(kernel_api_demo_exit);
MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("syl");
