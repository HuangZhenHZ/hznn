#include <bits/stdc++.h>
#include "hznn_old.cpp"
#include "hznn.h"
#include "hznn2.h"

hznn2 nn;

void init_nn(){
	const int sz[3]={28*28,500,10};
	nn.init(3,sz);
	//nn.init(28*28,500,10,500);
}

const int N=60000, in_sz=28*28, out_sz=10;
double in[N][in_sz], out[N][out_sz];
unsigned char corlab[N];

void read_in(){
	FILE *file = fopen("train-images-idx3-ubyte","rb");
	unsigned char buf[in_sz];
	int head[4];
	assert( fread(head,16,1,file)>0 );
	/*
	printf("head[0]=%d\n",head[0]);
	printf("head[1]=%d\n",head[1]);
	printf("head[2]=%d\n",head[2]);
	printf("head[3]=%d\n",head[3]);
	assert( head[1]==60000 );
	assert( head[2]==28 );
	assert( head[3]==28 );
	*/

	for(int i=0; i<N; ++i){
		assert( fread(buf,in_sz,1,file)>0 );
		for(int j=0; j<in_sz; ++j){
			in[i][j] = buf[j]/255.0;
		}
	}
}

void read_out(){
	FILE *file = fopen("train-labels-idx1-ubyte","rb");
	//unsigned char buf[N];
	int head[2];
	assert( fread(head,8,1,file)>0 );
	//printf("head[0]=%d\n",head[0]);
	//printf("head[1]=%d\n",head[1]);
	//assert( head[1]==60000 );
	assert( fread(corlab,N,1,file)>0 );

	for(int i=0; i<N; ++i){
		for(int j=0; j<out_sz; ++j){
			out[i][j] = 0;
		}
		out[i][corlab[i]] = 1;
	}
}

void train(){
	double fout[out_sz];
	int cnt=0;

	double sum=0, num=0;

	for(int i=0; i<N; ++i){
		sum += nn.calc(in[i]);
		num += 1;
		nn.get_output(fout);

		int lab=0;
		for(int j=1; j<9; ++j){
			if (fout[j]>fout[lab]){
				lab=j;
			}
		}

		cnt += (lab==corlab[i]);

		nn.bp(out[i]);

		if (i%5000==0){
			printf("i=%d sum/num=%lf\n",i,sum/num);
			sum=0; num=0;
		}
	}

	printf("cnt=%d\n",cnt);
}

int main(){
	read_in();
	read_out();
	init_nn();

	/*
	for(int i=0; i<N; ++i){
		printf("corlab[%d]=%d\n",i,corlab[i]);
	}
	*/

	for(int t=0; t<1; ++t){
		printf("t=%d\n",t);
		train();
	}
	return 0;
}

