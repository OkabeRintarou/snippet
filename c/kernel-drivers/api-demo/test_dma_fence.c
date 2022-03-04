#include <asm/fcntl.h>
#include <asm/ioctl.h>
#include <linux/dma-fence.h>
#include <linux/file.h>
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/sync_file.h>
#include "test_dma_fence.h"

#define DMA_FENCE_IN_CMD		_IOWR('f', 0, int)
#define DMA_FENCE_OUT_CMD		_IOWR('f', 1, int)
#define DMA_FENCE_SIGNAL_CMD	_IO('f', 2)

static int in_fence_fd = -1;
static int out_fence_fd = -1;

struct dma_fence *out_fence = NULL;
struct dma_fence_cb cb;

static DEFINE_SPINLOCK(fence_lock);

static void dma_fence_cb(struct dma_fence *fence, struct dma_fence_cb *cb) {
	printk(KERN_INFO "dma-fence callbacl\n");
}

static const char *dma_fence_get_name(struct dma_fence *fence) {
	return "dma-fence-example";
}

static const struct dma_fence_ops fence_ops = {
	.get_driver_name = dma_fence_get_name,
	.get_timeline_name = dma_fence_get_name,
};

static struct dma_fence *create_fence(void) {
	struct dma_fence *fence;

	fence = kzalloc(sizeof(*fence), GFP_KERNEL);
	if (!fence)
		return NULL;

	dma_fence_init(fence, &fence_ops, &fence_lock, 0, 0);

	return fence;
}

static int fence_open(struct inode *inode, struct file *filp) {
	out_fence = create_fence();
	if (!out_fence)
		return -ENOMEM;
	return 0;
}

static int fence_close(struct inode *inode, struct file *filp) {
	dma_fence_put(out_fence);
	return 0;
}

static long fence_ioctl(struct file *filp,
		unsigned cmd, unsigned long arg) {

	struct sync_file *sync_file;
	struct dma_fence *in_fence;

	switch (cmd) {
	case DMA_FENCE_SIGNAL_CMD:
		if (out_fence) {
			printk(KERN_INFO "Signal Fence\n");
			dma_fence_signal(out_fence);
		}
		break;
	case DMA_FENCE_IN_CMD:
		if (copy_from_user(&in_fence_fd, (void __user*)arg, sizeof(int)) != 0)
			return -EFAULT;
		printk(KERN_INFO "Get in-fence from user: fd = %d\n", in_fence_fd);
		in_fence = sync_file_get_fence(in_fence_fd);

		if (!in_fence)
			return -EINVAL;

		dma_fence_add_callback(in_fence, &cb, dma_fence_cb);
		printk(KERN_INFO "Waiting in-fence to be signaled, process is blocking...\n");
		dma_fence_wait(in_fence, true);
		printk(KERN_INFO "in-fence signaled, process exit\n");

		dma_fence_put(in_fence);
		break;
	case DMA_FENCE_OUT_CMD:
		if (!out_fence)
			return -EINVAL;

		sync_file = sync_file_create(out_fence);
		out_fence_fd = get_unused_fd_flags(O_CLOEXEC);
		fd_install(out_fence_fd, sync_file->file);
	
		if (copy_to_user((void __user*)arg, &out_fence_fd, sizeof(int)) != 0)
			return -EFAULT;

		printk(KERN_INFO "Created an out-fence: fd = %d\n", out_fence_fd);

		dma_fence_put(out_fence);
		break;
	default:
		printk(KERN_INFO "Unknow cmd: %d\n", cmd);
		break;
	}
	return 0;
}

static struct file_operations fence_fops = {
	.owner = THIS_MODULE,
	.unlocked_ioctl = fence_ioctl,
	.open = fence_open,
	.release = fence_close,
};

static struct miscdevice mdev = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "dma-fence",
	.fops = &fence_fops,
};

int test_dma_fence_init(void) {
	return misc_register(&mdev);
}

void test_dma_fence_fini(void) {
	misc_deregister(&mdev);
}
