//#####################################################################
// N-Body Simulation. Authors Eitan Vilker, 
// To compile: run nvcc -std=c++11 DataSetDriverNBody.cu -o NBody
// To visualize: ./opengl_viewer.exe -m fluid -o output
//#####################################################################

#ifndef __DataSetDriverNBody_h__
#define __DataSetDriverNBody_h__
#include "../io_adapter/io_adapter.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <string>
#include <wchar.h>
#ifdef WIN32
#include <windows.h>
#elif defined(__linux__)
#include <sys/stat.h>
#endif

#ifdef WIN32
#define NO_MINMAX
#undef min
#undef max
#endif

using namespace std;

#define block_size 64

template<class T_VAL> void Write_Binary(std::ostream& output,const T_VAL& data)
{output.write(reinterpret_cast<const char*>(&data),sizeof(T_VAL));}

template<class T_VAL> void Write_Binary_Array(std::ostream& output,const T_VAL* array,const int n)
{if(n>0)output.write(reinterpret_cast<const char*>(array),n*sizeof(T_VAL));}

template<class T_VAL> bool Write_Binary_To_File(const std::string& file_name,const T_VAL& data)
{std::ofstream output(file_name,std::ios::binary);if(!output)return false;Write_Binary(output,data);return true;}

template<class T_VAL> bool Write_Binary_Array_To_File(const std::string& file_name,T_VAL* array,const int n)
{std::ofstream output(file_name,std::ios::binary);if(!output)return false;Write_Binary_Array(output,array,n);return true;}

template<class T_VAL> void Write_Text(std::ostream& output,const T_VAL& data)
{output<<data;}

template<class T_VAL> bool Write_Text_To_File(const std::string& file_name,const T_VAL& data)
{std::ofstream output(file_name);if(!output)return false;Write_Text(output,data);return true;}

#ifdef WIN32
inline bool Directory_Exists(const char* dirname)
{DWORD attr=GetFileAttributes(dirname);return((attr!=-1)&&(attr&FILE_ATTRIBUTE_DIRECTORY));}

inline bool Create_Directory(const std::string& dirname)
{
    if(!Directory_Exists(dirname.c_str())){size_t pos=0;
        do{pos=dirname.find_first_of("\\/",pos+1);
        if(!Directory_Exists(dirname.substr(0,pos).c_str())){
            if(CreateDirectory(dirname.substr(0,pos).c_str(),NULL)==0 && ERROR_ALREADY_EXISTS!=GetLastError()){
                std::cerr<<"Error: [File] Create directory "<<dirname<<" failed!"<<std::endl;return false;}}}while(pos!=std::string::npos);}
    return true;
}
#elif defined(__linux__)
inline bool Directory_Exists(const char* dirname)
{struct stat s;return stat(dirname,&s)==0;}

inline bool Create_Directory(const std::string& dirname)
{if(!Directory_Exists(dirname.c_str())){size_t pos=0;
        do{pos=dirname.find_first_of("\\/",pos+1);
            if(!Directory_Exists(dirname.substr(0,pos).c_str())){
                if(mkdir(dirname.substr(0,pos).c_str(),0755)!=0 && errno!=EEXIST){
                    std::cerr<<"Error: [File] Create directory "<<dirname<<"failed!"<<std::endl;return false;}}}
		while(pos!=std::string::npos);}
return true;}
#endif

//////////////////////////////////////////////////////////////////////////
////IO functions
//////////////////////////////////////////////////////////////////////////

std::string Frame_Dir(const std::string& output_dir,const int frame)
{
	return output_dir+"/"+std::to_string(frame);
}

void Create_Folder(const std::string& output_dir,const int frame)
{
	if(!Directory_Exists(output_dir.c_str()))Create_Directory(output_dir);

	std::string frame_dir=Frame_Dir(output_dir,frame);
	if(!Directory_Exists(frame_dir.c_str()))Create_Directory(frame_dir);
		
	{std::string file_name=output_dir+"/0/last_frame.txt";
	Write_Text_To_File(file_name,std::to_string(frame));}
}

template<class T> void Write_Grid(const std::string& file_name,const int* cell_counts,const T* dx,const T* domain_min)
{
    std::ofstream output(file_name,std::ios::binary);
	if(!output){std::cerr<<"Cannot open file "<<file_name<<std::endl;return;}

	Write_Binary_Array(output,cell_counts,3);
    Write_Binary(output,*dx);
    Write_Binary_Array(output,domain_min,3);
}

template<class T> void Write_Points(const std::string& file_name,const T* x,const int n)
{
	std::ofstream output(file_name,std::ios::binary);
	if(!output){std::cerr<<"Cannot open file "<<file_name<<std::endl;return;}

	Write_Binary(output,n);
	if(n>0){Write_Binary_Array(output,x,n*3);}
}

template<class T> void Write_Particles(const std::string& file_name,double4* x,const unsigned int n,const T* v=nullptr,const T* f=nullptr,const T* m=nullptr)
{
	std::ofstream output(file_name,std::ios::binary);
	if(!output){std::cerr<<"Cannot open file "<<file_name<<std::endl;return;}

	Write_Binary(output,n);
	if(n==0)return;

	std::vector<T> placeholder_T(n*3,(T)0);
	std::vector<int> placeholder_i(n,0);

	if(n>0){Write_Binary_Array(output,x,n*3);}

	if(v)Write_Binary_Array(output,v,n*3);
	else Write_Binary_Array(output,&placeholder_T[0],n*3);

	if(f)Write_Binary_Array(output,f,n*3);
	else Write_Binary_Array(output,&placeholder_T[0],n*3);

	if(m)Write_Binary_Array(output,m,n);
	else Write_Binary_Array(output,&placeholder_T[0],n);

	Write_Binary_Array(output,&placeholder_T[0],n);

	Write_Binary_Array(output,&placeholder_i[0],n);
}

template<class T> void Write_Scalar_Field(const std::string& file_name,const T* s,const int* counts)
{
	std::ofstream output(file_name,std::ios::binary);
	if(!output){std::cerr<<"Cannot open file "<<file_name<<std::endl;return;}

	Write_Binary_Array(output,counts,3);

	int n=counts[0]*counts[1]*counts[2];
	Write_Binary_Array(output,s,n);
}

template<class T> void Write_Vector_Field(const std::string& file_name,const T* v,const int* counts)
{
	std::ofstream output(file_name,std::ios::binary);
	if(!output){std::cerr<<"Cannot open file "<<file_name<<std::endl;return;}

	Write_Binary_Array(output,counts,3);

	int n=counts[0]*counts[1]*counts[2]*3;
	Write_Binary_Array(output,v,n);
}


/********************* Sums up the accelerations for each tile **************************/
__device__ double3 Tile_Calculation(double4 myPosition, double4 currentPosition, double3 acceleration, const double epsilon_squared) {

	double3 r;
	// r_ij
	r.x = currentPosition.x - myPosition.x;
	r.y = currentPosition.y - myPosition.y;
	r.z = currentPosition.z - myPosition.z;

	// distSqr = dot(r_ij, r_ij) + EPS^2
	double distSqr = r.x * r.x + r.y * r.y + r.z * r.z + epsilon_squared;

	// invDistCube = 1 / pow(distSqr, 3/2);
	double distSixth = distSqr * distSqr * distSqr;
	double invDistCube = 1.0f / sqrtf(distSixth);

	// s = m_j * invDistCube
	double s = currentPosition.w * invDistCube;

	// a_i = a_i + s * r_ij
	acceleration.x += r.x * s;
	acceleration.y += r.y * s;
	acceleration.z += r.z * s;

	return acceleration;
}


/******************** Computes forces and sets positions for each particle ****************************/
__global__ void Calculate_Forces(double4* device_position, double4* device_acceleration, double4* device_velocity, unsigned int n, double dt, const double epsilon_squared){
		
	extern __shared__ double4 shPosition[];

	double4 myPosition;
	double3 acc = { 0.0f, 0.0f, 0.0f };
	int i, tile;
	int gtid = blockIdx.x * blockDim.x + threadIdx.x;
	myPosition = device_position[gtid];
	int p = 16; // p is the number of threads

	//if (gtid == gridDim.x * blockDim.x / 2) printf("pos: %.10f, %.10f, %.10f\n", device_position[gtid].x, device_position[gtid].y, device_position[gtid].z);
	

	for (i = 0, tile = 0; i < n; i += p, tile++) {
		int idx = tile * blockDim.x + threadIdx.x;
		shPosition[threadIdx.x] = device_position[idx];
		__syncthreads();
		
		#pragma unroll 32
		for(int j = 0; j < block_size; j++) {

			acc = Tile_Calculation(myPosition, shPosition[j], acc, epsilon_squared);
			__syncthreads();
		}
	}

	/****************** Save the result in global memory for the integration step *****************/
		
	double4 acc4 = { acc.x, acc.y, acc.z, 0.0f };
	device_acceleration[gtid] = acc4;

	device_velocity[gtid].x += device_acceleration[gtid].x * dt;
	device_velocity[gtid].y += device_acceleration[gtid].y * dt;
	device_velocity[gtid].z += device_acceleration[gtid].z * dt;

	device_position[gtid].x += device_velocity[gtid].x * dt;
	device_position[gtid].y += device_velocity[gtid].y * dt;
	device_position[gtid].z += device_velocity[gtid].z * dt;

}

int main(){

	int simulation_num = 1;    // Determines which simulation is being run

	/************** Variables initialized ***************/
	unsigned int grid_size = 0;
	unsigned int n = 0;
	double dt = 0.0;
	double epsilon = 0.0;
	double epsilon_squared = 0.0;
	int frame_total = 0;
	unsigned int block_num = 0;
	double dx = 0.0;
		
	double* mass;							////mass array

	double* positions_x;
	double* positions_y;
	double* positions_z;

	double4* GPU_positions;
	double4* GPU_velocities;
	double4* GPU_accelerations;

	double3* Write_Particle_Positions = new double3[n];


	/************* Parameters to simulation set *******************/
	switch(simulation_num){
	
		case 1:

			// Variables to adjust here
			grid_size = 16;
			n = grid_size * grid_size * grid_size; // number of particles
			dt = 0.001;
			epsilon = 1e-2;
			epsilon_squared = epsilon * epsilon;
			frame_total = 10;
			block_num = n / block_size;
		
			mass = new double[n];								////mass array

			positions_x = new double[n];
			positions_y = new double[n];
			positions_z = new double[n];

			GPU_positions = new double4[n];
			GPU_velocities = new double4[n];
			GPU_accelerations = new double4[n];
			
			//// Initialize particles as the cell centers on a background grid
			dx = 1.0/grid_size;
			for (unsigned int k = 0; k < grid_size; k++) {
			
				for (unsigned int j = 0; j < grid_size; j++) {
				
					for (unsigned int i = 0; i < grid_size; i++) {
					
						unsigned int index = k * grid_size * grid_size + j * grid_size + i;

						// Initialize position
						positions_x[index] = dx * (double)i;
						positions_y[index] = dx * (double)j;
						positions_z[index] = dx * (double)k;

						GPU_positions[index] = { positions_x[index], positions_y[index], positions_z[index], 0.0 };
						GPU_velocities[index] = { 0.0, 0.0, 0.0, 0.0 };
						GPU_accelerations[index] = { 0.0, 0.0, 0.0, 0.0 };

					}
				}
			}

			for (int i = 0; i < n; i++) {
				mass[i] = 100.0;
				GPU_positions[i].w = mass[i];
			}

			break;

		/***************** Solar System Model **********************/
		case 2:

			// Variables to adjust here
			n = 11; // number of particles
			dt = 100;
			epsilon = 1e-2;
			epsilon_squared = epsilon * epsilon;
			frame_total = 100;
			block_num = n / block_size;
		
			mass = new double[n];								////mass array

			positions_x = new double[n];
			positions_y = new double[n];
			positions_z = new double[n];

			GPU_positions = new double4[n];
			GPU_velocities = new double4[n];
			GPU_accelerations = new double4[n];


			// Sun
			positions_x[0] = 0;
			positions_y[0] = 0;
			positions_z[0] = 0;

			GPU_positions[0] = { positions_x[0], positions_y[0], positions_z[0], 0.0 };
			GPU_velocities[0] = { 0.0, 0.0, 0.0, 0.0 };
			GPU_accelerations[0] = { 0.0, 0.0, 0.0, 0.0 };
	
			mass[0] = 1.989e30;
			GPU_positions[0].w = mass[0];

			// Mercury
			positions_x[1] = -57.9e6;
			positions_y[1] = 0;
			positions_z[1] = 0;

			GPU_positions[1] = { positions_x[1], positions_y[1], positions_z[1], 0.0 };
			GPU_velocities[1] = { 0.0, 47.4e3, 0.0, 0.0 };
			GPU_accelerations[1] = { 0.0, 0.0, 0.0, 0.0 };

			mass[1] = 0.33e24;
			GPU_positions[1].w = mass[1];

			// Venus
			positions_x[2] = -108.2e6;
			positions_y[2] = 0;
			positions_z[2] = 0;

			GPU_positions[2] = { positions_x[2], positions_y[2], positions_z[2], 0.0 };
			GPU_velocities[2] = { 0.0, 35.0e3, 0.0, 0.0 };
			GPU_accelerations[2] = { 0.0, 0.0, 0.0, 0.0 };

			mass[2] = 4.87e24;
			GPU_positions[2].w = mass[2];

			// Earth
			positions_x[3] = -149.6e6;
			positions_y[3] = 0;
			positions_z[3] = 0;

			GPU_positions[3] = { positions_x[3], positions_y[3], positions_z[3], 0.0 };
			GPU_velocities[3] = { 0.0, 29.8e3, 0.0, 0.0 };
			GPU_accelerations[3] = { 0.0, 0.0, 0.0, 0.0 };

			mass[3] = 5.97e24;
			GPU_positions[3].w = mass[3];

			// Moon
			positions_x[4] = positions_x[3] - 0.384e6; // Position relative to Earth
			positions_y[4] = 0;
			positions_z[4] = 0;

			GPU_positions[4] = { positions_x[4], positions_y[4], positions_z[4], 0.0 };
			GPU_velocities[4] = { 0.0, 24.1e3, 0.0, 0.0 };
			GPU_accelerations[4] = { 0.0, 0.0, 0.0, 0.0 };
	
			mass[4] = 0.073e24;
			GPU_positions[4].w = mass[4];

			// Mars
			positions_x[5] = -227.9e6;
			positions_y[5] = 0;
			positions_z[5] = 0;
				 
			GPU_positions[5] = { positions_x[5], positions_y[5], positions_z[5], 0.0 };
			GPU_velocities[5] = { 0.0, 24.1e3, 0.0, 0.0 };
			GPU_accelerations[5] = { 0.0, 0.0, 0.0, 0.0 };
	
			mass[5] = 0.642e24;
			GPU_positions[5].w = mass[5];

			// Jupiter
			positions_x[6] = -778.6e6;
			positions_y[6] = 0;
			positions_z[6] = 0;
				 
			GPU_positions[6] = { positions_x[6], positions_y[6], positions_z[6], 0.0 };
			GPU_velocities[6] = { 0.0, 13.1e3, 0.0, 0.0 };
			GPU_accelerations[6] = { 0.0, 0.0, 0.0, 0.0 };
	
			mass[6] = 1898e24;
			GPU_positions[6].w = mass[6];

			// Saturn
			positions_x[7] = -1433.5e6;
			positions_y[7] = 0;
			positions_z[7] = 0;
				 
			GPU_positions[7] = { positions_x[7], positions_y[7], positions_z[7], 0.0 };
			GPU_velocities[7] = { 0.0, 9.7e3, 0.0, 0.0 };
			GPU_accelerations[7] = { 0.0, 0.0, 0.0, 0.0 };
	
			mass[7] = 568e24;
			GPU_positions[7].w = mass[7];

			// Uranus
			positions_x[8] = -2872.5e6;
			positions_y[8] = 0;
			positions_z[8] = 0;
				 
			GPU_positions[8] = { positions_x[8], positions_y[8], positions_z[8], 0.0 };
			GPU_velocities[8] = { 0.0, 6.8e3, 0.0, 0.0 };
			GPU_accelerations[8] = { 0.0, 0.0, 0.0, 0.0 };
	
			mass[8] = 568e24;
			GPU_positions[8].w = mass[8];

			// Neptune
			positions_x[9] = -4495.1e6;
			positions_y[9] = 0;
			positions_z[9] = 0;
				 
			GPU_positions[9] = { positions_x[9], positions_y[9], positions_z[9], 0.0 };
			GPU_velocities[9] = { 0.0, 5.4e3, 0.0, 0.0 };
			GPU_accelerations[9] = { 0.0, 0.0, 0.0, 0.0 };
	
			mass[9] = 102e24;
			GPU_positions[9].w = mass[9];

			// Pluto
			positions_x[10] = -5906.4e6;
			positions_y[10] = 0;
			positions_z[10] = 0;
				 
			GPU_positions[10] = { positions_x[10], positions_y[10], positions_z[10], 0.0 };
			GPU_velocities[10] = { 0.0, 4.7e3, 0.0, 0.0 };
			GPU_accelerations[10] = { 0.0, 0.0, 0.0, 0.0 };
	
			mass[10] = 0.0146e24;
			GPU_positions[10].w = mass[10];

			break;

		default:

			cout<<"Improper case entered"<<endl;
	}

	/************* Memory allocated ****************/

	double* velocities_x = new double[n];
	memset(velocities_x, 0x00, n * sizeof(double));
	double* velocities_y = new double[n];
	memset(velocities_y, 0x00, n * sizeof(double));
	double* velocities_z = new double[n];
	memset(velocities_z, 0x00, n * sizeof(double));

	double* accelerations_x = new double[n];
	memset(accelerations_x, 0x00, n * sizeof(double));
	double* accelerations_y = new double[n];
	memset(accelerations_y, 0x00, n * sizeof(double));
	double* accelerations_z = new double[n];
	memset(accelerations_z, 0x00, n * sizeof(double));


	/*
	// CUDA timing
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time = 0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);
	*/

	//// allocate arrays on device
	double4* device_position = nullptr;
	double4* device_velocity = nullptr;
	double4* device_acceleration = nullptr;

	cudaMalloc((void**)&device_position, 4 * n * sizeof(double));
	cudaMalloc((void**)&device_velocity, 4 * n * sizeof(double));
	cudaMalloc((void**)&device_acceleration, 4 * n * sizeof(double));

	cudaMemcpy(device_position, GPU_positions, 4 * n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(device_velocity, GPU_velocities, 4 * n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(device_acceleration, GPU_accelerations, 4 * n * sizeof(double), cudaMemcpyHostToDevice);

	int isPython = 0;

	/*********** Simulation is executed **************/

	// First frame

	std::string output_dir;
	std::string frame_dir;
	std::string file_name;

	if (isPython == 1){
		output_dir="py_output";
		Create_Directory(output_dir.c_str());
	}
	else{
		output_dir="output";
		frame_dir=Frame_Dir(output_dir,0);
		Create_Folder(output_dir,0);
		file_name=frame_dir+"/particles";
		Write_Particles<double>(file_name,GPU_positions,n);
	}

	std::string out_file_name;
	ofstream out_file;

	if(isPython == 1){
		
		out_file_name = output_dir;
		out_file_name += "/";
		out_file_name += to_string(0) + ".txt";
		out_file.open(out_file_name);

		for (int i = 0; i < n; i++){

			std::string current_line = std::to_string(GPU_positions[i].x);
			current_line += ",";
			current_line += std::to_string(GPU_positions[i].y);
			current_line += ",";
			current_line += std::to_string(GPU_positions[i].z);
			current_line += "\n";
			out_file << current_line;
		}
	
		out_file.close();
	}
	
	for(int frame = 1; frame < frame_total  + 1; frame++){

		if(isPython == 1){
			out_file_name += "/";
			out_file_name += to_string(frame) + ".txt";
			out_file.open(out_file_name);
		}

		if(isPython == 0){
			frame_dir=Frame_Dir(output_dir,frame);
			Create_Folder(output_dir,frame);
		}

		Calculate_Forces<<<block_num, block_size, 4 * block_size * sizeof(double)>>> (device_position, device_velocity, device_acceleration, n, dt, epsilon_squared);

		cudaMemcpy(GPU_positions, device_position, 4 * n * sizeof(double), cudaMemcpyDeviceToHost);

		// Change positions from double4 to double3
		
		if(isPython == 1){
			
			for (int i = 0; i < n; i++){

				std::string current_line = std::to_string(GPU_positions[i].x);
				current_line += ",";
				current_line += std::to_string(GPU_positions[i].y);
				current_line += ",";
				current_line += std::to_string(GPU_positions[i].z);
				current_line += "\n";
				out_file << current_line;
			}

			out_file.close();
		}

		if(isPython == 0){

			std::string file_name=frame_dir+"/particles";
			Write_Particles<double>(file_name,GPU_positions,n);
		}

		//printf("%f, %f, %f,\n", Write_Particle_Positions[n/2].x, Write_Particle_Positions[n/2].y, Write_Particle_Positions[n/2].z);

	}


	/*********** Memory freed *************/

	cudaFree(device_position);
	cudaFree(device_velocity);
	cudaFree(device_acceleration);


	/*
	// CUDA timing
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time, start, end);
	printf("\nGPU runtime: %.4f ms\n", gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	*/

	return 0;
}

#endif
