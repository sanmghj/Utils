/*
    cudnn test program
    gcc -o test_cudnn test_cudnn.c -I(*cudnn include 경로) -L(*cudnn lib 경로)
    ./test_cudnn
*/
#include <cudnn.h>
#include <stdio.h>

int main(void)
{
    cudnnHandle_t handle;
    cudnnStatus_t status = cudnnCreate(&handle);
    if (status != CUDNN_STATUS_SUCCESS) {
        printf("Error creating cuDNN handle: %s\n", cudnnGetErrorString(status));
        return 1;
    }

    printf("cuDNN handle created successfully.\n");

    // Clean up
    cudnnDestroy(handle);
    return 0;
}