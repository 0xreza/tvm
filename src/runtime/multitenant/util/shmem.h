#include <string>
#include <unistd.h>
#include <sys/syscall.h>
#include <iostream>
#include <sys/stat.h>
#include <fcntl.h>
#include <sstream>
#include <fstream>


// Wrapper to call memfd_create syscall
static inline int memfd_create(const char *name, unsigned int flags) {
	return syscall(__NR_memfd_create, name, flags);
}

inline std::string shm_file_path(int fd) {
	std::stringstream ss;
	ss << "/proc/" << getpid() << "/fd/" << fd;
	return ss.str();
}

// Creates an empty in-memory file
inline std::string create_memfile() {
	int shm_fd = memfd_create(std::string("").c_str(), 0);
	if (shm_fd < 0) { //Something went wrong :(
		fprintf(stderr, "[- Could not open shm file descriptor\n");
		exit(-1);
	}
	return shm_file_path(shm_fd);
}

/**
Creates an in-memory file and copies the contents of the specified file to it.
Returns the filename of the newly created in-memory file.
Exits on error (hey, this isn't supposed to be robust yet)
**/
inline std::string copy_to_memory(std::string filename) {
	std::string memfile = create_memfile();
	std::ifstream src(filename, std::ios::binary);
    std::ofstream dst(memfile,   std::ios::binary);

    dst << src.rdbuf();
    return memfile;
}
