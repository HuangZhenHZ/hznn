#include <bits/stdc++.h>
#include "hznn.h"

inline void hznn::wdata::init_rand(){
	w = ( rand() % 2001 - 1000 ) / 5000.0;
	s = 0;
}

inline void hznn::wdata::grad(double dx){
	s += dx * dx;
	w -= dx / sqrt(s + 0.5) * 0.05;
}

void hznn::init(int _in, int _mid, int _out, int _pre){
	in=_in; mid=_mid; out=_out; pre=_pre;
	int sum=0;
	for(int i = in; i < in+mid+out; ++i)
		sum += i;

	wdata *pool = new wdata[sum];

	for(int i = in; i < in+mid+out; ++i){
		//w[i] = new wdata[i];
		w[i] = pool;
		pool += i;
		//memset(w[i], 0, sizeof(wdata)*i);
		//c1[i] = c2[i] = (wdata){double(0.001),double(0)};
		for(int j=0; j<i; ++j) w[i][j].init_rand();
		c1[i].init_rand();
		c2[i].init_rand();
	}
}

int hznn::calc(const double f[]){
	memcpy(fout,f,sizeof(double)*in);
	top = 0;
	for(int i=0; i<in; ++i)
		if (fout[i]>0)
			sta[top++] = i;

	int in_top = top;
/*
	for(int i=in; i<in+pre; ++i){
		in1[i] = c1[i].w;
		for(int k=0; k<top && sta[k]<in; ++k)
			in1[i] += fout[sta[k]] * w[i][sta[k]].w;

		fout[i] = in1[i]>0 ? in1[i] : 0;
		if (in1[i]>0) sta[top++] = i;
	}
*/
	for(int i=in; i<in+pre; ++i) in1[i] = c1[i].w;

	for(int k=0; k<top && sta[k]<in; ++k){
		int sk = sta[k];
		for(int i=in; i<in+pre; ++i)
			in1[i] += fout[sk] * w[i][sk].w;
	}

	for(int i=in; i<in+pre; ++i){
		fout[i] = in1[i]>0 ? in1[i] : 0;
		if (in1[i]>0) sta[top++] = i;
	}

	for(int i=in+pre; i<in+mid; ++i){
		assert(false);

		in1[i] = c1[i].w;
		int k = top-1;
		while (k>=0 && sta[k]>=i-pre){
			in1[i] += fout[sta[k]] * w[i][sta[k]].w;
			k--;
		}

		if (in1[i]<=0){
			fout[i] = 0;
			continue;
		}

		in2[i] = c2[i].w;
		while (k>=0){
			in2[i] += fout[sta[k]] * w[i][sta[k]].w;
			k--;
		}

		for(k=top-1; k>=0 && sta[k]>=i-pre; k--)
			w[i][sta[k]].grad(fout[sta[k]]*(in1[i]-in2[i]));

		c1[i].grad(in1[i]-in2[i]);

		if (in2[i]<=0){
			fout[i] = 0;
			continue;
		}

		//fout[i] = in1[i] * in2[i];
		fout[i] = in2[i];
		sta[top++] = i;
	}

	for(int i=in+mid; i<in+mid+out; ++i){
		in1[i] = c1[i].w;
		for(int k=0; k<top; ++k)
			in1[i] += fout[sta[k]] * w[i][sta[k]].w;

		fout[i] = 1.0 / ( 1.0 + exp(-in1[i]) );
	}

	return top - in_top;
}

void hznn::get_output(double f[]){
	memcpy(f,fout+in+mid,sizeof(double)*out);
}

void hznn::bp(const double f[]){
	//for(int i=0; i<in+mid; ++i) d[i]=0;
	for(int k=0; k<top; ++k) d[sta[k]]=0;

	for(int i=in+mid; i<in+mid+out; ++i){
		d[i] = fout[i] - f[i-(in+mid)];
		d[i] *= fout[i] * (1 - fout[i]);
		c1[i].grad(d[i]);

		for(int k=0; k<top; ++k){
			d[sta[k]] += d[i] * w[i][sta[k]].w;
			w[i][sta[k]].grad(fout[sta[k]]*d[i]);
		}
	}

	while (top && sta[top-1]>=in+pre){
		assert(false);

		int i=sta[--top];
		c1[i].grad(d[i] * in2[i]);
		c2[i].grad(d[i] * in1[i]);

		int k = top-1;
		while (k>=0 && sta[k]>=i-pre){
			d[sta[k]] += d[i] * w[i][sta[k]].w * in2[i];
			w[i][sta[k]].grad(fout[sta[k]]*d[i]*in2[i]);
			k--;
		}

		while (k>=0){
			d[sta[k]] += d[i] * w[i][sta[k]].w * in1[i];
			w[i][sta[k]].grad(fout[sta[k]]*d[i]*in1[i]);
			k--;
		}
	}

	while (top && sta[top-1]>=in){
		int i=sta[--top];
		c1[i].grad(d[i]);

		for(int k=0; k<top && sta[k]<in; ++k){
			d[sta[k]] += d[i] * w[i][sta[k]].w;
			w[i][sta[k]].grad(fout[sta[k]]*d[i]);
		}
	}
}
