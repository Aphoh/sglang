#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

int cr_plugin_dump_file(int fd, int id)
{
	struct stat statbuf;

	(void)id;
	if (fstat(fd, &statbuf) != 0)
		return -1;
	if (!S_ISCHR(statbuf.st_mode) || major(statbuf.st_rdev) != 195 ||
	    minor(statbuf.st_rdev) != 255)
		return -ENOTSUP;
	return 0;
}

int cr_plugin_restore_file(int id)
{
	(void)id;
	return open("/dev/nvidiactl", O_RDWR);
}
