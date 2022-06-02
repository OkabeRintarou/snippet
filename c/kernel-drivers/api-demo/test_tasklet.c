#include <linux/interrupt.h>
#include "test_tasklet.h"

static const char tasklet_data[] = "We use a string; but it could be a pointer"
                                   " to a structure";

void tasklet_work(struct tasklet_struct *t) {
  pr_info("%s\n", (const char *)(t->data));
}

DECLARE_TASKLET(my_tasklet, tasklet_work);

void test_tasklet_init(void) {
  my_tasklet.data = (unsigned long)(tasklet_data);
  tasklet_schedule(&my_tasklet);
}

void test_tasklet_fini(void) {
  tasklet_kill(&my_tasklet);
}
