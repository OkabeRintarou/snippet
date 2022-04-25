#include "test_container_of.h"
#include <linux/kernel.h>
#include <linux/printk.h>

struct Person {
  int age;
  int salary;
  char *name;
};

static void test_container_of_0(int *age) {
  pr_warn("get person from age: 0x%llx\n",
          (unsigned long long)(container_of(age, struct Person, age)));
}

static void test_container_of_1(int *salary) {
  pr_warn("get person from salary: 0x%llx\n",
          (unsigned long long)(container_of(salary, struct Person, salary)));
}

static void test_container_of_2(char **name) {
  pr_warn("get person from name: 0x%llx\n",
          (unsigned long long)(container_of(name, struct Person, name)));
}

void test_container_of(void) {
  struct Person person;
  test_container_of_0(&person.age);
  test_container_of_1(&person.salary);
  test_container_of_2(&person.name);
}