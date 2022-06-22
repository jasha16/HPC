#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <time.h>

#define N 3

void print_matrix(float m[N*(N+1)]){
    int i, j;
    for(i=0;i<N;i++){
        for(j=0;j<N+1;j++){
            printf("%.2f ", m[i * (N+1) + j]);
        }
        printf("\n");
    }

}

void print_result(float x[N]){
    int i;
    printf("Answer is: \n");

    for(i=0;i<N;i++){
        printf("X%d = %.2f ", i, x[i]);
    }

    printf("\n");

}

int main(void) {
    int pid, np;
    float matrix[N*(N+1)] = {1,1,1,10,2,5,4,25,7,2,10,68};
    float matrix_recv[N*(N+1)];
    int i, j, k, c, p, u;
    float ans[N] = {0}, ans_recv[N];
    float ratio;
    MPI_Status status;
    int start_recv, stop_recv, startbs_recv, stopbs_recv;
    double start_time, end_time;
    float partial_recv;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    //Master process
    if(pid == 0){
        printf("Initial Matrix: \n");
        print_matrix(matrix);
        printf("\n");

        for(i=0;i<N;i++){
            
            //Update matrix for sub-processes
            for(p=1;p<np;p++){
                MPI_Send(&matrix,N*(N+1),MPI_INT, p, i, MPI_COMM_WORLD);
            }

            int num_elems = (N-(i+1))/np;
            int start, stop;
            int r = (N-(i+1)) % np;

            if(r > 0){
                start = num_elems + 1 + i + 1;
            }else{
                start = num_elems + i + 1;
            }
            //send j values to sub-processes
            for(p=1;p<np-1;p++){

                if(p < r){
                    stop = start + num_elems + 1;
                }else{
                    stop = start + num_elems;
                }
                MPI_Send(&start,1,MPI_INT, p, i, MPI_COMM_WORLD);
                MPI_Send(&stop,1,MPI_INT, p, i, MPI_COMM_WORLD);
                start = stop;
            }

            //Last sub-process gets remaining tasks
            start = stop;
            stop = N;
            MPI_Send(&start,1,MPI_INT, p, i, MPI_COMM_WORLD);
            MPI_Send(&stop,1,MPI_INT, p, i, MPI_COMM_WORLD);

            //Calculate locally
            if(r > 0){
                stop = num_elems + 1 + i + 1;
            }else{
                stop = num_elems + i + 1;
            }
            
            for(j=i+1; j<stop; j++){
                ratio = matrix[j * (N+1) + i]/matrix[i * (N+1) + i];
                for(k=0;k<N+1;k++){
                    matrix[j * (N+1) + k] = matrix[j * (N+1) + k] - (ratio * matrix[i * (N+1) + k]);
                }
            }

            //Update matrix
            for(p=1; p<np; p++){
                MPI_Recv(&start_recv,1, MPI_INT, p, i,MPI_COMM_WORLD,&status);
                MPI_Recv(&stop_recv,1, MPI_INT, p, i,MPI_COMM_WORLD,&status);
                MPI_Recv(&matrix_recv,N*(N+1), MPI_INT, p, i,MPI_COMM_WORLD,&status);

                for(u=start_recv; u<stop_recv; u++){
                    for(k=0;k<N+1;k++){
                        matrix[j * (N+1) + k] = matrix_recv[j * (N+1) + k];
                    }
                }
            }
        }

        //After Elimination
        printf("After Elimination: \n");
        print_matrix(matrix);
        printf("\n");

        //Update sub-processes with matrix
        for(p=1;p<np;p++){
            MPI_Send(&matrix,N*(N+1),MPI_INT, p, i, MPI_COMM_WORLD);
        }

        //Back Substitution
        for(i=N-1; i>=0; i--){
            //Update matrix for sub-processes
            ans[i] = matrix[i * (N+1) + N];
            for(p=1;p<np;p++){
                MPI_Send(&ans,N,MPI_INT, p, i, MPI_COMM_WORLD);
            }
            int num_elems = (N-(i+1))/np;
            int start_bs, stop_bs;
            int r = (N-(i+1)) % np;

            if(r > 0){
                start_bs = num_elems + 1 + i + 1;
            }else{
                start_bs = num_elems + i + 1;
            }
            //send j values to sub-processes
            for(p=1;p<np-1;p++){

                if(p < r){
                    stop_bs = start_bs + num_elems + 1;
                }else{
                    stop_bs = start_bs + num_elems;
                }
                MPI_Send(&start_bs,1,MPI_INT, p, i, MPI_COMM_WORLD);
                MPI_Send(&stop_bs,1,MPI_INT, p, i, MPI_COMM_WORLD);
                start_bs = stop_bs;
            }
            //Last sub-process gets remaining tasks
            start_bs = stop_bs;
            stop_bs = N;
            MPI_Send(&start_bs,1,MPI_INT, p, i, MPI_COMM_WORLD);
            MPI_Send(&stop_bs,1,MPI_INT, p, i, MPI_COMM_WORLD);

            //Calculate locally
            if(r > 0){
                stop_bs = num_elems + 1 + i + 1;
            }else{
                stop_bs = num_elems + i + 1;
            }

            float sub = 0;
            for(j=i+1; j<stop_bs; j++){
                //ans[i] = ans[i] - matrix[i*(N+1) + j] * ans[j];
                sub += matrix[i*(N+1) + j] * ans[j];
            }

            //Recieve partial sums
            for(p=1; p<np; p++){
                MPI_Recv(&partial_recv,1, MPI_FLOAT, p, i,MPI_COMM_WORLD,&status);
                sub += partial_recv;
            }

            ans[i] = ans[i] - sub;

            ans[i] = ans[i]/matrix[i*(N+1) + i];
        }

        print_result(ans);
        printf("\n");
    }
       
    //Sub-process
    else{
        for(i=0;i<N;i++){
            //Update local copy of matrix
            MPI_Recv(&matrix_recv,N*(N+1), MPI_INT, 0, i,MPI_COMM_WORLD,&status);

            //Recieve start and stop j values
            MPI_Recv(&start_recv,1, MPI_INT, 0, i,MPI_COMM_WORLD,&status);
            MPI_Recv(&stop_recv,1, MPI_INT, 0, i,MPI_COMM_WORLD,&status);

            for(j=start_recv; j<stop_recv; j++){
                ratio = matrix[j * (N+1) + i]/matrix[i * (N+1) + i];
                for(k=0;k<N+1;k++){
                    matrix_recv[j * (N+1) + k] = matrix_recv[j * (N+1) + k] - (ratio * matrix_recv[i * (N+1) + k]);
                }
            }

            //Send results back to master
            MPI_Send(&start_recv,1,MPI_INT, 0, i, MPI_COMM_WORLD);
            MPI_Send(&stop_recv,1,MPI_INT, 0, i, MPI_COMM_WORLD);
            MPI_Send(&matrix_recv,N*(N+1),MPI_INT, 0, i, MPI_COMM_WORLD);
        }

        //Update local copy of matrix
        MPI_Recv(&matrix_recv,N*(N+1), MPI_INT, 0, i,MPI_COMM_WORLD,&status);

        //Back substitution
        for(i=N-1; i>=0; i--){
            //Update local copy of array
            MPI_Recv(&ans_recv,N, MPI_INT, 0, i,MPI_COMM_WORLD,&status);

            //Recieve start and stop j values
            MPI_Recv(&startbs_recv,1, MPI_INT, 0, i,MPI_COMM_WORLD,&status);
            MPI_Recv(&stopbs_recv,1, MPI_INT, 0, i,MPI_COMM_WORLD,&status);

            float partial = 0;
            for(j=startbs_recv; j<stopbs_recv; j++){
                partial += matrix_recv[i*(N+1) + j] * ans_recv[j];
            }

            //Send partial sum back
            MPI_Send(&partial,1,MPI_FLOAT, 0, i, MPI_COMM_WORLD);
        }
       
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    MPI_Finalize();

    if(pid == 0){
        printf("Runtime: %fs\n\n",end_time-start_time);
    }

    return 0;
}
