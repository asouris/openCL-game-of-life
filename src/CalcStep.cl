
int worldIdx(int i, int j, const int N, const int M){
    i = (i + N) % N;
    j = (j + M) % M;
	return j + i * M;
}

__kernel void calcStep(global int *current, global int *next, int N, int M){
    int gindex = get_global_id(0);
    int globaliOG = get_global_id(0), globaljOG = get_global_id(1);
    int threadsi = get_global_size(0), threadsj = get_global_size(1); //number of threads

    for(int globali = globaliOG; globali < N; globali += threadsi){
        for(int globalj = globaljOG; globalj < M; globalj += threadsj){
            //get number of neighbours
            int neighbours = current[worldIdx(globali - 1, globalj - 1, N, M)] + current[worldIdx(globali - 1, globalj, N, M)] + current[worldIdx(globali - 1, globalj + 1, N, M)] +
                            current[worldIdx(globali, globalj - 1, N, M)] + current[worldIdx(globali, globalj + 1, N, M)] +
                            current[worldIdx(globali + 1, globalj - 1, N, M)] + current[worldIdx(globali + 1, globalj, N, M)] + current[worldIdx(globali + 1, globalj + 1, N, M)];

            //set next step 
            next[worldIdx(globali, globalj, N, M)] = neighbours == 3 || (neighbours == 2 && current[worldIdx(globali, globalj, N, M)]);        
        }
    }

    

}