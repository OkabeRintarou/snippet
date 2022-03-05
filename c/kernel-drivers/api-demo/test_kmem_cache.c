#include <linux/printk.h>
#include <linux/slab.h>
#include "test_kmem_cache.h"

#define MAX_NAME 64

struct student {
	char name[MAX_NAME];
	int age;
	char sex;
};

void student_ctor(void *p) {
	struct student *pstu = (struct student *)p;

	pstu->name[0] = '\0';
	pstu->age = 0;
	pstu->sex = 0;
}

static struct kmem_cache *cache = NULL;
static struct student *students[10] = {NULL};

void test_kmem_cache_init(void) {
	int i;
	
	cache = kmem_cache_create(
				"student-mem-cache", sizeof(struct student), 
				0, SLAB_HWCACHE_ALIGN, student_ctor);
	if (!cache) {
		printk(KERN_INFO "not enough system physical memory\n");
		return;
	}

	for (i = 0; i < 10; i++) {
		students[i] = kmem_cache_alloc(cache, GFP_KERNEL);
		if (!students[i])
			break;
		else
			printk(KERN_INFO "Create student %d: %llx\n", i, (unsigned long long)students[i]);
	}
}

void test_kmem_cache_fini(void) {
	int i;

	for (i = 0; i < 10; i++) {
		if (!students[i])
			kmem_cache_free(cache, students[i]);
	}
	kmem_cache_destroy(cache); }
