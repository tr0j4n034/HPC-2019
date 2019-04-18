#include <stdio.h>
#include <math.h>

int main()
{
    const int N=200;
    const int M=200;
    
    const double Niter = 1000;
    
    size_t counter = 0;
    
 
    FILE * writefile;
    writefile=fopen("out.txt", "w");

    double T_new[N][M];
    double T_old[N][M];


    for(int i=0; i<N; i++)
    {
        T_old[i][0]=0.0;
        T_new[i][0]=0.0;
        T_old[i][M-1]=0.0;
        T_new[i][M-1]=0.0;
    }

    for(int j=0; j<M; j++)
    {
        T_old[0][j]=1.0;
        T_new[0][j]=1.0;
        T_old[N-1][j]=0.0;
        T_new[N-1][j]=0.0;
    }

    
    while (counter<Niter)
    {
        
        for(int i=1; i<N-1; i++)
        {
            for(int j=1; j<M-1; j++)
            {
                T_new[i][j]=0.25*(T_old[i+1][j]+T_old[i-1][j]+T_old[i][j+1]+T_old[i][j-1]);
            }
        }

        for(int i=1; i<N-1; i++)
        {
            for(int j=1; j<M-1; j++)
            {
                T_old[i][j]=0.25*(T_new[i+1][j]+T_new[i-1][j]+T_new[i][j+1]+T_new[i][j-1]);
                
            }
        }

        counter=counter+2;        
    }

    for(int i=0; i<N; i++)
    {
        for(int j=0; j<M; j++)
        {
            fprintf(writefile,"%e\t", T_old[i][j]);
        }
        fprintf(writefile, "\n");
    }

    fclose(writefile);
    return 0;
}

