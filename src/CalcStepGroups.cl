
int worldIdx(int i, int j, const int N, const int M){
    i = (i + N) % N;
    j = (j + M) % M;
	return j + i * M;
}
#define block_size 8

// world is 16 x 16
// group is 4 x 4
__kernel void calcStep(__global int *currentGlobal, __global int *next, int N, int M){
    //int gindex = get_global_id(0);
    //int numThreads = get_global_size(0); //number of threads
    
    // The cells on a corner of the group write 4 cells, themselfs and
    // the other 3 that touches it (outside of group area)
    // The cells in a border but not a corner write themselfs and 
    // the one that its touching (outside)
    // Any other cell in the interior of the group only writes themselfs.


    // position of thread in the group
    int igroup = get_local_id(0), jgroup = get_local_id(1);
    // size of group
    int nigroupOG = get_local_size(0), njgroupOG = get_local_size(1);
    // id of group
    int groupindexI = get_local_id(0), groupindexJ = get_local_id(1);

    //position on global world
    int globaliOG = get_global_id(0);
    int globaljOG = get_global_id(1);

    // the local buffer should be (2 + groupDim(0)) x (2 + groupDim(1)) 
    //const int block_size = (niloc + 2) * (2 + njloc);
    local int currentLocal[(block_size+2) * (block_size+2)];

    //position on local memory
    int iloc = igroup+1, jloc = jgroup+1;

    for(int globali = globaliOG; globali < N; globali += get_global_size(0)){
        for(int globalj = globaljOG; globalj < M; globalj += get_global_size(1)){

            //adjust group size for incomplete groups
            int offseti = max((globali + (nigroupOG - (igroup + 1))) - N + 1, 0);
            int offsetj = max((globalj + (njgroupOG - (jgroup + 1))) - M + 1, 0);
            int nigroup = nigroupOG - offseti, njgroup = njgroupOG - offsetj;

            // size of local memory
            int niloc = nigroup+2, njloc = njgroup+2;

    
            //copies itself to local memory
            currentLocal[worldIdx(iloc, jloc, niloc, njloc)] = currentGlobal[worldIdx(globali, globalj, N, M)];
            
            //number of extra copies
            int copies =  (igroup == 0 && jgroup == 0) +            //corners, just 1
                        (igroup == 0 && jgroup == njgroup-1) +       
                        (igroup == nigroup-1 && jgroup == 0) +
                        (igroup == nigroup-1 && jgroup == njgroup-1) +
                        (igroup == 0) +                         //borders, could be 1 or 2
                        (igroup == nigroup-1) +
                        (jgroup == 0) +
                        (jgroup == njgroup-1);
            
            // traslation on i and j
            int borderI = (igroup == 0) * -1 + (igroup == nigroup-1); 
            int borderJ = (jgroup == 0) * -1 + (jgroup == njgroup-1);

            // if copies == 3, will copy all this positions
            // otherwise, its just a border and will only copy the first one.
            int positions[6] = {borderI, borderJ, 
                            borderI, 0,
                            0, borderJ};


            for(int k = 0; k < copies*2; k+=2){
                currentLocal[worldIdx(iloc + positions[k], jloc + positions[k+1], niloc, njloc)] = currentGlobal[worldIdx(globali + positions[k], globalj + positions[k+1], N, M)];
            }

            // we wait everyone to make the copies
            barrier(CLK_LOCAL_MEM_FENCE);

            //Now the cell can access its 9 neighbors faster...
            //Since most of them only copied 1 cell... should be faster

            //get number of neighbours
            int neighbours = currentLocal[worldIdx(iloc - 1, jloc - 1, niloc, njloc)] + currentLocal[worldIdx(iloc - 1, jloc, niloc, njloc)] + currentLocal[worldIdx(iloc - 1, jloc + 1, niloc, njloc)] +
                            currentLocal[worldIdx(iloc, jloc - 1, niloc, njloc)] + currentLocal[worldIdx(iloc, jloc + 1, niloc, njloc)] +
                            currentLocal[worldIdx(iloc + 1, jloc - 1, niloc, njloc)] + currentLocal[worldIdx(iloc + 1, jloc, niloc, njloc)] + currentLocal[worldIdx(iloc + 1, jloc + 1, niloc, njloc)];

            //set next step

            //next[worldIdx(globali, globalj, N, M)] = currentGlobal[worldIdx(globali, globalj, N, M)];
            //next[worldIdx(get_global_id(0), get_global_id(1), N, M)] = currentGlobal[worldIdx(get_global_id(0), get_global_id(1), N, M)];
            //next[worldIdx(get_global_id(0), get_global_id(1), N, M)] = neighbours;
            next[worldIdx(globali, globalj, N, M)] = neighbours == 3 || (neighbours == 2 && currentLocal[worldIdx(iloc, jloc, niloc, njloc)]);
            //next[worldIdx(globali, globalj, N, M)] = borderI;
        }
    }

}