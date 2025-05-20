
int worldIdx(int i, int j, const int N, const int M){
    i = (i + N) % N;
    j = (j + M) % M;
	return j + i * M;
}

__kernel void calcStep(global int *current, global int *next, int N, int M){
    int gindex = get_global_id(0);
    int numThreads = get_global_size(0); //number of threads

    for(int index = gindex; index < N*M; index+=numThreads){
        //get indices
        int i = index / M; //row
        int j = index % M; //column


        //get number of neighbours
        int neighbours = current[worldIdx(i - 1, j - 1, N, M)] + current[worldIdx(i - 1, j, N, M)] + current[worldIdx(i - 1, j + 1, N, M)] +
                        current[worldIdx(i, j - 1, N, M)] + current[worldIdx(i, j + 1, N, M)] +
                        current[worldIdx(i + 1, j - 1, N, M)] + current[worldIdx(i + 1, j, N, M)] + current[worldIdx(i + 1, j + 1, N, M)];

        //set next step 
        next[index] = neighbours == 3 || (neighbours == 2 && current[index]);
    }

}