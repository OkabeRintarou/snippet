#include <cerrno>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

struct buffer_object {
	uint32_t width;
	uint32_t height;
	uint32_t pitch;
	uint32_t handle;
	uint32_t size;
	uint32_t *vaddr;
	uint32_t fb_id;
};

static void init(
		drmModeConnectorPtr *connector,
		drmModeFBPtr *fb,
		drmModeCrtcPtr *crtc,
		int *drm_fd,
		uint32_t *width,
		uint32_t *height) {

	int fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
	assert(fd >= 0);

	drmModeResPtr res = drmModeGetResources(fd);
	assert(res);

	for (int i = 0; i < res->count_connectors; i++) {
		(*connector) = drmModeGetConnector(fd, res->connectors[i]);
		assert(*connector);

		if ((*connector)->connection == DRM_MODE_CONNECTED)
			break;
		drmFree(*connector);
		*connector = nullptr;
	}
	assert(*connector);

	auto encoder = drmModeGetEncoder(fd, (*connector)->encoder_id);
	assert(encoder);

	*crtc = drmModeGetCrtc(fd, encoder->crtc_id);
	assert(*crtc);

	*fb = drmModeGetFB(fd, (*crtc)->buffer_id);
	assert(*fb);

	*drm_fd = fd;
	*width = (*connector)->modes[0].hdisplay;
	*height = (*connector)->modes[0].vdisplay;

	drmFree(encoder);
	drmFree(res);

}

static void destroy_fb(int fd, struct buffer_object *bo) {
	struct drm_mode_destroy_dumb destroy{};
	int r;

	drmModeRmFB(fd, bo->fb_id);
	munmap(bo->vaddr, bo->size);

	destroy.handle = bo->handle;
	r = drmIoctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy);
	assert(r == 0);
}

static void create_fb(int fd, struct buffer_object *bo, uint32_t color) {
	struct drm_mode_create_dumb create{};
	struct drm_mode_map_dumb map{};
	int r;

	create.width = bo->width;
	create.height = bo->height;
	create.bpp = 32;
	r = drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &create);
	assert(r == 0);

	// bind the dumb-buffer to an FB object
	bo->pitch = create.pitch;
	bo->size = create.size;
	bo->handle = create.handle;
	drmModeAddFB(fd, bo->width, bo->height, 24, 32, bo->pitch,
			bo->handle, &bo->fb_id);
	// map the dumb-buffer to userspace
	map.handle = create.handle;
	r = drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &map);
	assert(r == 0);

	bo->vaddr = (uint32_t*)mmap(0, create.size, PROT_READ | PROT_WRITE,
			MAP_SHARED, fd, map.offset);
	assert(bo->vaddr != nullptr);

	for (int i = 0; i < (bo->size / 4); i++)
		bo->vaddr[i] = color;
}

int main() {
	int fd;
	drmModeConnectorPtr connector;
	drmModeCrtcPtr crtc;
	drmModeFBPtr fb;
	uint32_t width, height;
	buffer_object bos[3];
	uint32_t colors[3] = {0xff0000, 0x00ff00, 0x0000ff};

	init(&connector, &fb, &crtc, &fd, 
			&width, &height);
	
	for (int i = 0; i < 3; i++) {
		bos[i].width = width;
		bos[i].height = height;
		create_fb(fd, &bos[i], colors[i]);
	}
	for (int i = 0; i < 3; i++) {
		drmModeSetCrtc(fd, crtc->crtc_id, bos[i].fb_id,
				0, 0, &connector->connector_id, 1, &connector->modes[0]);
		getchar();
	}

	// cleanup
	for (int i = 0; i < 3; i++)
		destroy_fb(fd, &bos[i]);
	drmModeFreeConnector(connector);
	close(fd);

	return 0;
}
