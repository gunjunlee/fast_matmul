#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <chrono>
#include <omp.h>
#include <fstream>

using namespace std;

#define SIZE 2048//matrix size

float A_[SIZE*SIZE];//for 1D
float B_[SIZE*SIZE];
float C_[SIZE*SIZE];

int parallel(int i) {
	__asm {

		push eax;//save eax
		push edx;//save edx
		push ebx;//save ebx
		push ecx;//save dcx

		mov eax, i;//eax=i

		xor edx, edx;//edx=j
	L2:

		push eax;//save eax
		imul eax, eax, 4*SIZE;//eax=eax*4*SIZE

		push edx;//save edx
		imul edx, edx, 4*SIZE;//edx=edx*4*SIZE

		lea ecx, A_;
		add eax, ecx;//eax=A_+i*SIZE
		lea ecx, B_;
		add edx, ecx;//edx=B_+j*SIZE
		lea ecx, C_;// ecx=C_
		mov ebx, [esp];//ebx=j
		imul ebx, ebx, 4;
		add ecx, ebx;
		mov ebx, [esp + 4];//ebx=i
		imul ebx, ebx, 4;
		imul ebx, ebx, SIZE;
		add ecx, ebx;//ecx=C_+i*SIZE+j

		xor ebx, ebx;//ebx=k;
	L3:

		movss xmm1, [eax];//xmm0=A_[i*SIZE+k]
		movss xmm2, [eax+4];//xmm2=A_[i*SIZE+k+1]
		movss xmm3, [eax + 8];//xmm2=A_[i*SIZE+k+2]
		movss xmm4, [eax + 12];//xmm2=A_[i*SIZE+k+3]
		movss xmm0, [ecx];//xmm1=C_[i*SIZE+j]


		mulss xmm1, [edx]; //A_[i*SIZE+k] * B_[j*SIZE + k]
		mulss xmm2, [edx+4]; //A_[i*SIZE+k+1] * B_[j*SIZE + k + 1]
		mulss xmm3, [edx + 8]; //A_[i*SIZE+k+2] * B_[j*SIZE + k + 2]
		mulss xmm4, [edx + 12]; //A_[i*SIZE+k+3] * B_[j*SIZE + k + 3]

		addss xmm1, xmm0;
		addss xmm2, xmm1;
		addss xmm3, xmm2;
		addss xmm4, xmm3;

		movss [ecx], xmm4;//C_[i*SIZE+j]=C_[i*SIZE+j] + ...

		add eax, 16;// eax=&A_[i*SIZE+k]->eax=&A_[i*SIZE+k+2]
		add edx, 16;// edx=&B_[j*SIZE+k]->edx=&B_[j*SIZE+k+2]

		add ebx, 4;//k=k+2;
		cmp ebx, SIZE;//k==SIZE?
		jne L3;

		pop edx;
		pop eax;
		add edx, 1;//edx=edx+1
		cmp edx, SIZE;
		jne L2;

		pop ecx;
		pop ebx;
		pop edx;
		pop eax;
	}
}

int main()
{
	srand(time(NULL));
	
	//-------------------Preprocessing--------------------//
	//make array
	float** A = new float*[SIZE];
	float** B = new float*[SIZE];
	float** C = new float*[SIZE];
	int i, j;
	int temp;
	
	ifstream fin;
	fin.open("A_2048.txt");//read A
	
	for (i = 0; i < SIZE; i++) {
		A[i] = new float[SIZE];
		B[i] = new float[SIZE];
		C[i] = new float[SIZE];
	}
	

	fin >> temp;
	fin >> temp;
	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			fin >> A[i][j];
		}
	}
	
	fin.close();

	fin.open("B_2048.txt");//read B

	fin >> temp;
	fin >> temp;
	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			fin >> B[i][j];
		}
	}

	fin.close();

	for (i = 0; i < SIZE; i++) {//make matrix to 1D
		for (j = 0; j < SIZE; j++) {
			A_[i*SIZE + j] = A[i][j];
		}
	}
	
	for (i = 0; i < SIZE; i++) {//make matrix to 1D
		for (j = 0; j < SIZE; j++) {
			B_[i + j*SIZE] = B[i][j];//transpose
		}
	}

	chrono::system_clock::time_point StartTime = chrono::system_clock::now();
	//---------------Matrix Multiplication---------------//
	omp_set_num_threads(4);//|thread|=4
	
	
#pragma omp parallel for//multithread
	for (i = 0; i < SIZE; i++) {//calculate for each i
		parallel(i);
	}
	
	//-------------------Preprocessing--------------------//

	chrono::system_clock::time_point EndTime = chrono::system_clock::now();
	chrono::microseconds micro = chrono::duration_cast<chrono::microseconds>(EndTime - StartTime);
	cout << "Matrix Multiplication done" << endl;
	cout << "Time : " << micro.count() << endl;
	
	for (i = 0; i < SIZE; i++) {//make 1D to 2D matrix
		for (j = 0; j < SIZE; j++) {
			C[i][j] = C_[i*SIZE + j];
		}
	}

	for (i = 0; i < SIZE; i++) {
		delete[] A[i];
		delete[] B[i];
		delete[] C[i];
	}

	delete[] A;
	delete[] B;
	delete[] C;
	
	system("PAUSE");

	return 0;
}