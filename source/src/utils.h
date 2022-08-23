#ifndef _UTILS_
#define _UTILS_

#include <cstdlib>

void fill_random_uint8(uint8_t*data, int m, int n, int sparsity) {
	for (int i = 0; i < m*n; i++) {
		data[i] = rand() % 100;
		if (data[i] < sparsity) //made sparse 
			data[i] = 0;
	}
}

void fill_random(float*data, int m, int n, float sparsity) {
	for (int i = 0; i < m*n; i++) {
		data[i] = rand()/double(RAND_MAX);
		if (data[i] < 0.01 * sparsity) //made sparse 
			data[i] = 0;
	}
}

void printMatrix(int m, int n, const float*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            float Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}

void printintMatrix(int m, int n, const int*A, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            int Areg = A[col + row*n];
            // printf("%s(%d,%d) = %d\n", name, row+1, col+1, Areg);
            std::cout << std::left << std::setw(4) << Areg;
        }
        std::cout << std::endl;
    }
}

template <typename type>
void printMatrix(int m, int n, const type*A, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            float Areg = A[col + row*n];
            // printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
            std::cout << std::left << std::setw(4) << Areg;
        }
        std::cout << std::endl;
    }
}

void _itoa(const unsigned long long int a, char *s)
{
    for (int i = 0; i < 64; i++)
    {
        if (((a >> i) & 1) == 0x01)
        {
            s[i] = '1';
        }
        else {s[i] = '0';}
    }
}

void _itoa_32(const int a, char *s)
{
    for (int i = 0; i < 32; i++)
    {
        if (((a >> i) & 1) == 0x01)
        {
            s[i] = '1';
        }
        else {s[i] = '0';}
    }
}

void printlongintMatrix(int m, const unsigned long long int*A, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        unsigned long long int Areg = A[row];
        char s[64];
        _itoa(Areg, s);
        // printf("%s(%d,%d) = %s\n", name, row+1, col+1, s);
        for (int i = 0; i < 64; i++) {
            std::cout << s[i];
        }
        std::cout << std::endl;
    }
}

void printintMatrix_32(int m, const int*A, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        int Areg = A[row];
        char s[32];
        _itoa_32(Areg, s);
        // printf("%s(%d,%d) = %s\n", name, row+1, col+1, s);
        for (int i = 0; i < 32; i++) {
            std::cout << s[i];
        }
        std::cout << "  value: " << Areg << std::endl;
    }
}

#endif