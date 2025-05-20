#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstdarg>
#include <map>

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#define THREADS 50
#define BLOCK_SIZE 10

using std::chrono::microseconds;


// Queue abstraction del aux
class Queue
{
private:
    cl::Platform _platform;
    cl::Device _device;
    cl::Context _context;
    cl::CommandQueue _queue;
    std::vector<cl::Buffer> _buffers;
    cl::Kernel _kernel;
    cl::Program _program;
    void setKernelArgs(int idx) {} // Base case for recursion

    template <typename Last>
    void setKernelArgs(int idx, Last last)
    {
        _kernel.setArg(idx, last);
    };

    template <typename First, typename... Rest>
    void setKernelArgs(int idx, First first, Rest... rest)
    {
        _kernel.setArg(idx, first);
        setKernelArgs(idx + 1, rest...);
    };

public:
    Queue()
    {
        // Query for platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        _platform = platforms.front();
        std::cout << "Platform: " << _platform.getInfo<CL_PLATFORM_NAME>()
                  << std::endl;

        // Get a list of devices on this platform
        std::vector<cl::Device> devices;
        // Select the platform.
        _platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        _device = devices.front();
        std::cout << "Device: " << _device.getInfo<CL_DEVICE_NAME>()
                  << std::endl;

        // Create a context
        _context = cl::Context(devices);

        // Create a command queue
        // Select the device.
        _queue = cl::CommandQueue(_context, _device);
    }

    // Manda a la cola una escritura de buffer
    template <typename T>
    int addBuffer(std::vector<T> &data, cl_mem_flags flags = CL_MEM_READ_WRITE)
    {
        cl::Buffer buffer(_context, flags, data.size() * sizeof(T));
        _queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, data.size() * sizeof(T),
                                  data.data());
        _buffers.push_back(buffer);
        return _buffers.size() - 1;
    }

    template <typename T>
    void updateBuffer(std::vector<T> &data, int index){
        _queue.enqueueWriteBuffer(_buffers[index], CL_TRUE, 0, data.size() * sizeof(T),
                                  data.data());
    }

    // Lee el kernel de un archivo
    void setKernel(const std::string &file, const std::string &kernelName)
    {
        std::ifstream sourceFile(file);
        std::stringstream sourceCode;
        sourceCode << sourceFile.rdbuf();

        // Make and build program from the source code
        _program = cl::Program(_context, sourceCode.str(), true);

        // Make kernel
        _kernel = cl::Kernel(_program, kernelName.c_str());
    }

    // Lee a data el buffer #index
    template <typename T>
    void readBuffer(std::vector<T> &data, int index = 0)
    {
        _queue.enqueueReadBuffer(_buffers[index], CL_TRUE, 0,
                                 data.size() * sizeof(T), data.data());
    }


    template <typename... Args>
    cl::Event operator()(cl::NDRange globalSize, cl::NDRange localSize, Args... args)
    {
        // Set the kernel arguments
        for (size_t i = 0; i < _buffers.size(); ++i)
        {
            _kernel.setArg(i, _buffers[i]);
        }
        setKernelArgs(_buffers.size(), args...);

        cl::Event event;
        _queue.enqueueNDRangeKernel(_kernel, cl::NullRange, globalSize, localSize,
                                    nullptr, &event);
        event.wait();
        return event;
    }
};

void initWorld(std::vector<int> &world, const int M){
    const std::vector<std::pair<int,int>> glider = {
		{4, 4},
        {5, 5},
        {6, 3}, {6, 4}, {6, 5}
	};
    for (auto [i, j] : glider)
		world[j + i * M] = 1;
}

void printWorld(std::vector < int > &world, int N, int M){
    for(int i = 0; i < N*M; i++){
        if(world[i]) std::cout << "■";
        else std::cout << "□";
        if((i+1) % M == 0) std::cout << std::endl;
    }
    std::cout << std::endl;
}

void report(std::string rep){
    std::cout << rep << std::endl;
}


int main(int argc, char const *argv[])
{
    if (argc != 4)
	{
        report("usage: ./openclConway {N} {M} {steps}");
		exit(1);
	}

	int N = atoi(argv[1]);
	int M = atoi(argv[2]);


    try
    {

        Queue q;
        report("Queue created");

        auto t_start = std::chrono::high_resolution_clock::now();

        // empty world and next state
        std::vector<int> dCurrent(N*M, 0), dNext(N*M);
        //fill world
        initWorld(dCurrent, M);
        auto t_end = std::chrono::high_resolution_clock::now();
        auto t_create_data =
            std::chrono::duration_cast<microseconds>(t_end - t_start).count();

        report("World created:");
        printWorld(dCurrent, N, M);

        // Copy values from host variables to device
        t_start = std::chrono::high_resolution_clock::now();
        q.addBuffer(dCurrent, CL_MEM_READ_ONLY);
        q.addBuffer(dNext, CL_MEM_WRITE_ONLY);
        t_end = std::chrono::high_resolution_clock::now();
        auto t_copy_to_device =
            std::chrono::duration_cast<microseconds>(t_end - t_start).count();
        report("Values copied to device");

        // Read the program source
        q.setKernel("CalcStep.cl", "calcStep");
        report("Kernel sent to device");

        // Execute the function on the device 
        cl::NDRange globalSize(THREADS);
        cl::NDRange localSize(BLOCK_SIZE);

        t_start = std::chrono::high_resolution_clock::now();

        report("Starting cycle");
        
        const int maxStep = atoi(argv[3]);
        int step = 0;
        while (step++ <= maxStep){
        
            // calls kernel, the buffers are passed as first arguments in the order they were added
            // N and M are passed as the rest of the kernel arguments
            cl::Event event = q(globalSize, localSize, N, M);
            event.wait();

            // Copy the output variable from device to host
            // Copies it to current, for next step
            q.readBuffer(dNext, 1);
            printWorld(dNext, N, M);

            //write data in dNext to buffer index 0 (dCurrent)
            q.updateBuffer(dNext, 0);
        }

        report("Finished cycle");

        t_end = std::chrono::high_resolution_clock::now();
        auto t_kernel =
            std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
                .count();

        report("Time to create data: " + std::to_string(t_create_data) + " microseconds");
        report("Time to copy data to device: " + std::to_string(t_copy_to_device) + " microseconds");
        report("Time to execute kernel: " + std::to_string(t_kernel) + " microseconds");
        

    }
    catch (cl::Error err)
    {
        std::cerr << "Error (" << err.err() << "): " << err.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
    return 0;
}