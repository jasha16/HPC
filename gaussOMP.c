#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>

#define N 3

void print_matrix(float m[N][N+1]){
  int i, j;
  for(i=0;i<N;i++){
    for(j=0;j<N+1;j++){
      if(m[i][j] < 0){
        printf("%.2f  ", m[i][j]);
      }
      else{
        printf(" %.2f  ", m[i][j]);
      }
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

int main(void){
  
  float matrix[N][N+1] = {1,1,1,10,2,5,4,25,7,2,10,68};
  int i, j, k;
  float ans[N] = {0};
  float ratio;
  double start_time, end_time;
  
  printf("Initital \n");
  print_matrix(matrix);
  printf("\n");

  start_time = omp_get_wtime();
  
  //Gaussian Elimination
  for(i=0;i<N;i++){
    #pragma omp parallel shared(matrix,i,ratio) private(j,k)
    {
      #pragma omp for
      for(j=i+1;j<N;j++){

        #pragma omp parallel shared(matrix,i,j,ratio) private(k)
        {
          ratio = matrix[j][i] / matrix[i][i];
            #pragma omp for
            for(k=0;k<N+1;k++){
              matrix[j][k] = matrix[j][k] - (ratio * matrix[i][k]);
            }
        }
      }
    }
  }
  
  printf("\nAfter Elimination\n");
  print_matrix(matrix);
  
  //Back Substitution
  for(i=N-1; i>=0; i--){
    ans[i] = matrix[i][N];
      #pragma omp parallel for shared(ans, i, matrix) private(j)
      for(j=i+1; j<N; j++){
        ans[i] = ans[i] - matrix[i][j] * ans[j];
      }
      ans[i] = ans[i]/matrix[i][i];
    }
  
  end_time = omp_get_wtime();
  print_result(ans);
  printf("\nRuntime: %fs\n\n",end_time-start_time);
  
  return 0;
}

