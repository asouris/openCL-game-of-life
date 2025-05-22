# Conway's Game of Life on GPU - OpenCL
An assignment for the course of Computing on GPU.

```
mkdir build
cd build
cmake .. 
make
```

To run conway's game:
```
./bin/openclConway {grid size X} {grid size Y} {steps} {mode 0|1}
```
The mode parameter is to set the group optimization.