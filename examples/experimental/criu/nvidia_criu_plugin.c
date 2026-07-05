// SPDX-License-Identifier: Apache-2.0

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

static int is_nvidia_device(const struct stat *statbuf)
{
	return S_ISCHR(statbuf->st_mode) && major(statbuf->st_rdev) == 195;
}

int cr_plugin_dump_file(int fd, int id)
{
	struct stat statbuf;

	(void)id;
	if (fstat(fd, &statbuf) != 0)
		return -1;
	if (!is_nvidia_device(&statbuf) || minor(statbuf.st_rdev) != 255)
		return -ENOTSUP;
	return 0;
}

int cr_plugin_restore_file(int id)
{
	(void)id;
	return open("/dev/nvidiactl", O_RDWR);
}

int cr_plugin_handle_device_vma(int fd, const struct stat *statbuf)
{
	(void)fd;
	return is_nvidia_device(statbuf) ? 0 : -ENOTSUP;
}
