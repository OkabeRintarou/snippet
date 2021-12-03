#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>

struct modeset_dev {

	modeset_dev() {
		next = nullptr;
		saved_crtc = nullptr;
		map = nullptr;
		width = height = stride = size = 0;
		fb = conn = crtc = 0;
	}

	struct modeset_dev *next;

	uint32_t width;
	uint32_t height;
	uint32_t stride;
	uint32_t size;
	uint32_t handle;
	uint8_t *map;

	drmModeModeInfo mode;
	uint32_t fb;
	uint32_t conn;
	uint32_t crtc;
	drmModeCrtc *saved_crtc;
};

static modeset_dev *modeset_list = nullptr;

static int modeset_open(int *out, const char *node) {
	int ret = 0;
	int fd = open(node, O_RDWR | O_CLOEXEC);
	if (fd < 0) {
		ret = -errno;
		fprintf(stderr, "cannot open '%s': %m\n", node);
		return ret;
	}

	uint64_t has_dumb;
	if (drmGetCap(fd, DRM_CAP_DUMB_BUFFER, &has_dumb) < 0 || !has_dumb) {
		fprintf(stderr, "drm device '%s' does not support dumb buffers\n", node);
		close(fd);
		return -EOPNOTSUPP;
	}
	*out = fd;
	return 0;
}

static int modeset_create_fb(int fd, struct modeset_dev *dev) {
	struct drm_mode_create_dumb creq;
	struct drm_mode_destroy_dumb dreq;
	struct drm_mode_map_dumb mreq;
	int ret;

	// create dumb buffer
	memset(&creq, 0, sizeof(creq));
	creq.width = dev->width;
	creq.height = dev->height;
	creq.bpp = 32;
	ret = drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &creq);
	if (ret < 0) {
		fprintf(stderr, "cannot create dumb buffer (%d): %m\n", errno);
		return -errno;
	}

	dev->stride = creq.pitch;
	dev->size = creq.size;
	dev->handle = creq.handle;

	// create framebuffer object for the dumb-buffer
	ret = drmModeAddFB(fd, dev->width, dev->height, 24, 32, dev->stride, dev->handle, &dev->fb);
	if (ret) {
		fprintf(stderr, "cannot create framebuffer (%d):%m\n", errno);
		ret = -errno;
		goto err_destroy;
	}

	// prepare actual memory mapping
	memset(&mreq, 0, sizeof(mreq));
	mreq.handle = dev->handle;
	ret = drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &mreq);
	if (ret) {
		fprintf(stderr, "cannot map dumb buffer (%d): %m\n", errno);
		ret = -errno;
		goto err_fb;
	}
	dev->map = static_cast<uint8_t*>(
			mmap(0, dev->size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mreq.offset));

	if (dev->map == MAP_FAILED) {
		fprintf(stderr, "cannot mmap dumb buffer (%d): %m\n", errno);
		ret = -errno;
		goto err_fb;
	}
	// clear the framebuffer to 0
	memset(dev->map, 0, dev->size);
	return 0;
err_fb:
	drmModeRmFB(fd, dev->fb);
err_destroy:
	memset(&dreq, 0, sizeof(dreq));
	dreq.handle = dev->handle;
	drmIoctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &dreq);
	return ret;
}

static int modeset_find_crtc(int fd, drmModeRes *res, drmModeConnector *conn, modeset_dev *dev) {
	drmModeEncoder *enc = nullptr;
	int32_t crtc;
	
	if (conn->encoder_id) {
		enc = drmModeGetEncoder(fd, conn->encoder_id);
	}

	if (enc) {
		if (enc->crtc_id) {
			crtc = enc->crtc_id;
			for (auto iter = modeset_list; iter; iter = iter->next) {
				if (iter->crtc == crtc) {
					crtc = -1;
					break;
				}
			}
			if (crtc >= 0) {
				drmModeFreeEncoder(enc);
				dev->crtc = crtc;
				return 0;
			}
		}
		drmModeFreeEncoder(enc);
	}

	for (int i = 0; i < conn->count_encoders; i++) {
		enc = drmModeGetEncoder(fd, conn->encoders[i]);
		if (enc == nullptr) {
			fprintf(stderr, "cannot retrieve encoder %u:%u (%d): %m\n", i, conn->encoders[i], errno);
			continue;
		}

		for (int j = 0; j < res->count_crtcs; j++) {
			if (!(enc->possible_crtcs & (1 << j)))
				continue;

			crtc = res->crtcs[j];
			for (auto iter = modeset_list; iter; iter = iter->next) {
				if (iter->crtc == crtc) {
					crtc = -1;
					break;
				}
			}
			if (crtc >= 0) {
				drmModeFreeEncoder(enc);
				dev->crtc = crtc;
				return 0;
			}
		}
		drmModeFreeEncoder(enc);
	}
	fprintf(stderr, "cannot find suitable CRTC for connector %u\n", conn->connector_id);
	return -ENOENT;
}

static int modeset_setup_dev(int fd, drmModeRes *res, drmModeConnector *conn, modeset_dev *dev) {
	int ret;
	
	if (conn->connection != DRM_MODE_CONNECTED) {
		fprintf(stderr, "ignoring unused connector %u\n", conn->connector_id);
		return -ENOENT;
	}

	if (conn->count_modes == 0) {
		fprintf(stderr, "no valid mode for connector %u\n", conn->connector_id);
		return -EFAULT;
	}

	memcpy(&dev->mode, &conn->modes[0], sizeof(dev->mode));
	dev->width = conn->modes[0].hdisplay;
	dev->height= conn->modes[0].vdisplay;
	fprintf(stderr, "mode for connector %u is %ux%u\n", conn->connector_id, dev->width, dev->height);

	// find a crtc for this connector
	ret = modeset_find_crtc(fd, res, conn, dev);
	if (ret) {
		fprintf(stderr, "no valid crtc for connector %u\n", conn->connector_id);
		return ret;
	}

	// create a framebuffer for this CRTC
	ret = modeset_create_fb(fd, dev);
	if (ret) {
		fprintf(stderr, "cannot create framebuffer for connector %u\n", conn->connector_id);
		return ret;
	}
	return 0;

}

static int modeset_prepare(int fd) {

	// retrieve resources
	drmModeRes *res = drmModeGetResources(fd);
	if (res == nullptr) {
		fprintf(stderr, "cannot retrieve DRM resources (%d): %m\n", errno);
		return -errno;
	}
	// iterate all connectos
	for (int i = 0; i < res->count_connectors; i++) {
		drmModeConnector *conn = drmModeGetConnector(fd, res->connectors[i]);
		if (conn == nullptr) {
			fprintf(stderr, "cannot retrieve DRM connector %u:%u (%d): %m\n", i, res->connectors[i], errno);
			continue;
		}

		modeset_dev *dev = new modeset_dev();
		dev->conn = conn->connector_id;

		int ret = modeset_setup_dev(fd, res, conn, dev);
		if (ret) {
			if (ret != -ENOENT) {
				errno = -ret;
				fprintf(stderr, "cannot setup device for connector %u:%u (%d): %m\n", i, res->connectors[i], errno);
			}
			delete dev;
			drmModeFreeConnector(conn);
			continue;
		}
		drmModeFreeConnector(conn);
		dev->next = modeset_list;
		modeset_list = dev;
	}
	drmModeFreeResources(res);
	return 0;
}


static uint8_t next_color(bool *up, uint8_t cur, unsigned mod) {
	uint8_t next;

	next = cur + (*up ? 1 : -1) * (rand() % mod);
	if ((*up && next < cur) || (!*up && next > cur)) {
		*up = !*up;
		next = cur;
	}
	return next;
}

static void modeset_draw(void) {
	uint8_t r, g, b;
	bool r_up, g_up, b_up;
	unsigned offset;

	srand(time(nullptr));
	r = rand() % 0xff;
	g = rand() % 0xff;
	b = rand() % 0xff;
	r_up = g_up = b_up = true;

	for (int i = 0; i < 50; i++) {
		r = next_color(&r_up, r, 20);
		g = next_color(&g_up, g, 10);
		b = next_color(&b_up, b, 5);

		for (auto iter = modeset_list; iter; iter = iter->next) {
			for (int j = 0; j < iter->height; j++) {
				for (int k = 0; k < iter->width; k++) {
					offset = iter->stride * j + k * 4;
					*(uint32_t*)&iter->map[offset] =
						(r << 16) | (g << 8) | b;
					
				}
			}
		}

		usleep(100000);
	}
}

static void modeset_cleanup(int fd) {
	struct modeset_dev *iter;
	struct drm_mode_destroy_dumb dreq;

	while (modeset_list) {
		iter = modeset_list;
		modeset_list = iter->next;

		drmModeSetCrtc(fd,
				iter->saved_crtc->crtc_id,
				iter->saved_crtc->buffer_id,
				iter->saved_crtc->x,
				iter->saved_crtc->y,
				&iter->conn,
				1,
				&iter->saved_crtc->mode);
		drmModeFreeCrtc(iter->saved_crtc);

		munmap(iter->map, iter->size);

		drmModeRmFB(fd, iter->fb);

		memset(&dreq, 0, sizeof(dreq));
		dreq.handle = iter->handle;
		drmIoctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &dreq);

		free(iter);
	}
}

int main(int argc, char *argv[]) {
	int ret, fd;
	const char *card;

	if (argc > 1) {
		card = argv[1];
	} else {
		card = "/dev/dri/card0";
	}
	fprintf(stderr, "using card '%s'\n", card);

	// open the DRM device
	ret = modeset_open(&fd, card);
	if (ret)
		goto out_return;

	// prepare all connectos and CRTCs
	ret = modeset_prepare(fd);
	if (ret)
		goto out_close;

	for (auto iter = modeset_list; iter; iter = iter->next) {
		iter->saved_crtc = drmModeGetCrtc(fd, iter->crtc);
		ret = drmModeSetCrtc(fd, iter->crtc, iter->fb, 0, 0, &iter->conn, 1, &iter->mode);
		if (ret) {
			fprintf(stderr, "cannot set CRTC for connector %u (%d): %m\n", iter->conn, errno);
		}
	}

	// draw some colors for 5 seconds
	modeset_draw();

	// cleanup everything
	modeset_cleanup(fd);

	ret = 0;
out_close:
	close(fd);
out_return:
	if (ret) {
		errno = -ret;
		fprintf(stderr, "modeset failed with error %d: %m\n", errno);
	} else {
		fprintf(stderr, "exiting\n");
	}
	return ret;
}
