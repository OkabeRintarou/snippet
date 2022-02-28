#include <linux/list.h>
#include <linux/gfp.h>
#include <linux/slab.h>
#include "test_list.h"

struct fox {
	int number;
	struct list_head list;
};

static struct fox*
alloc_fox(int n) {
	struct fox *f;
	f = kmalloc(sizeof(*f), GFP_KERNEL);
	f->number = n;
	INIT_LIST_HEAD(&f->list);
	return f;
}

static void
alloc_foxs(unsigned total, struct list_head *h) {
	unsigned i;
	struct fox *cur;

	for (i = 0u; i < total; i++) {
		cur = alloc_fox(i);
		list_add(&cur->list, h);
	}
}

static void
alloc_foxs_tail(unsigned total, struct list_head *h) {
	unsigned i;
	struct fox *cur;

	for (i = 0u; i < total; i++) {
		cur = alloc_fox(i);
		list_add_tail(&cur->list, h);
	}
}

static void
print_foxs(struct list_head *h) {
	struct fox *f;

	list_for_each_entry(f, h, list) {
		printk(KERN_CONT "%d ", f->number);
	}

	printk(KERN_INFO "\n");
}

static void
free_foxs(struct list_head *h) {
	struct fox *cur, *next;

	list_for_each_entry_safe(cur, next, h, list) {
		list_del(&cur->list);
		kfree(cur);
	}
}

static void 
test_list_insert_and_remove(void) {
	LIST_HEAD(fox_list);
	alloc_foxs(5, &fox_list);
	alloc_foxs_tail(5, &fox_list);
	print_foxs(&fox_list);
	free_foxs(&fox_list);
	printk(KERN_INFO "fox list is empty: %d\n", list_empty(&fox_list));
}

static void 
test_list_splice(void) {
	LIST_HEAD(list1);
	LIST_HEAD(list2);

	alloc_foxs(5, &list1);
	alloc_foxs(5, &list2);
	list_splice(&list1, &list2);
	print_foxs(&list2);

	printk(KERN_INFO "list1 empty? %s, list2 empty? %s\n", 
		list_empty(&list1) ? "true" : "false",
		list_empty(&list2) ? "true" : "false");
}

void test_list(void) {
	test_list_insert_and_remove();
	test_list_splice();
}
