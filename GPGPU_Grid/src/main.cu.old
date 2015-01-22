#include <cuda.h>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>

using namespace std;

// Calcul C = A * B
__global__ void matrixMultiply(float * A, float * B, float * C,
        int numARows, int numAColumns,
        int numBRows, int numBColumns,
        int numCRows, int numCColumns) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= numCColumns || y >= numCRows) return;

    float s = 0.0;
    for(int z = 0; z<numAColumns; ++z)
        s += A[z+numAColumns*y] * B[x+numBColumns*z];
    C[x + numCColumns*y] = s;
}

#define gpuCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct Matrix
{
    string name;
    float * data;
    float * data_cuda;
    int dimx;
    int dimy;

    Matrix()
    {
        data = NULL;
        data_cuda = NULL;
    }

    void print()
    {
        cout<<"------"<<endl;
        cout<<name<<endl;
        cout<<dimx<<" x "<<dimy<<endl;
    }
    int size() const
    {
        return dimx*dimy;
    }
    void cudaAllocAndCpy()
    {
        gpuCheck( cudaMalloc((void**)&data_cuda,size()*sizeof(float)) );
        gpuCheck( cudaMemcpy(data_cuda,data,size()*sizeof(float),cudaMemcpyHostToDevice) );
    }

    void cudaGet()
    {
        gpuCheck( cudaMemcpy(data,data_cuda,size()*sizeof(float),cudaMemcpyDeviceToHost) );
    }

    void compare(const Matrix& other)
    {
        cout<<"comparaison"<<endl;
        cout<<dimx<<"|"<<other.dimx<<endl;
        cout<<dimy<<"|"<<other.dimy<<endl;
        int s = size();
        if (other.size() < s) s = other.size();
        s = std::min(s,10);
        cout << "size = "<<s<<endl;
        for(int i = 0; i<s ; ++i)
        {
            cout<<data[i]<<"|"<<other.data[i]<<endl;
        }
    }

    ~Matrix()
    {
        delete data;
    }
};

Matrix readMatrix(const string& file_string)
{
    cout<<"opening : "<<file_string<<endl;
    Matrix m;

    ifstream file;

    file.open(file_string.c_str());
    if (file)
    {
        stringstream dimension;
        string dimension_string;
        int dimx, dimy;

        getline(file,dimension_string);

        dimension << dimension_string;
        dimension >> dimx >> dimy;

        // alocation of the matrix
        m.name = file_string;
        m.dimx = dimx;
        m.dimy = dimy;
        m.data = new float[dimx*dimy];

        // read the file
        for(int y = 0 ; y < dimy ; ++y)
        {
            string line;
            stringstream s;
            getline(file,line);
            s << line;
            for(int x = 0 ; x < dimx ; ++x)
            {
                s >> m.data[x+dimx*y];
            }
        }


        file.close();
    }
    else
    {
        cerr<<"can't open the file "<<file_string<<endl;
    }
    return m;
}

void getMaxCudaThread(int& x, int&y, int&z)
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	for (int dev = 0; dev < deviceCount; dev++) {
		cudaDeviceProp deviceProp;

		cudaGetDeviceProperties(&deviceProp, dev);
        x = deviceProp.maxThreadsDim[0];
        y = deviceProp.maxThreadsDim[1];
        z = deviceProp.maxThreadsDim[2];
	}
}

int main(int argc, char ** argv) {

    /// Charger le fichier d'entree
    string data_dir = "./mp2_data/1";
    string input0_file = data_dir+"/input0.raw";
    string input1_file = data_dir+"/input1.raw";
    string output_file = data_dir+"/output.raw";

    Matrix input0 = readMatrix(input0_file);
    Matrix input1 = readMatrix(input1_file);
    Matrix output = readMatrix(output_file);
    Matrix output_real = readMatrix(output_file);

    /// afficher dimensions matrix
    input0.print();
    input1.print();
    output.print();

    /// Allouer la memoire sur GPU
    input0.cudaAllocAndCpy();
    input1.cudaAllocAndCpy();
    output.cudaAllocAndCpy();

    /// Initialiser la grille et les dimensions de chaque bloc
    int threadX,threadY,threadZ;

    getMaxCudaThread(threadX,threadY,threadZ);

    /*threadX = 8;*/
    /*threadY = 8;*/

    cout << "thread(" << threadX << "," << threadY << "," << threadZ << ")" << endl;
    int blockX = (output.dimx+threadX-1)/threadX;
    int blockY = (output.dimy+threadY-1)/threadY;

    dim3 block(blockX,blockY,1), thread(threadX,threadY,1);

    /// Execute le kernel
    matrixMultiply<<<thread,block>>>(
        input0.data_cuda,
        input1.data_cuda,
        output.data_cuda,
        input0.dimy, input0.dimx,
        input1.dimy, input1.dimx,
        output.dimy, output.dimx
    );

    gpuCheck(cudaThreadSynchronize());

    /// Charge le resultat en memoire CPU
    output.cudaGet();


    // test égalité
    output.compare(output_real);

    /// Libere la memoire
    return 0;
}

