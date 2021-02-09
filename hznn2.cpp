#include <bits/stdc++.h>
#include "hznn2.h"

void hznn2::LAY::init(int _n, int _tp){
	n = _n;
	tp = _tp;
	in = new double[n];
	out = new double[n];
	d = new double[n];
}

void hznn2::LAY::calc_out(){
	if (tp==0){
		for(int i=0; i<n; ++i)
			out[i] = in[i]>0 ? in[i] : 0;
		return;
	}
	if (tp==1){
		for(int i=0; i<n; ++i)
			out[i] = 1.0 / ( 1.0 + exp(-in[i]) );
		return;
	}
	assert(false);
}

void hznn2::LAY::bp_d(){
	if (tp==0){
		for(int i=0; i<n; ++i)
			if (in[i]<0) d[i]=0;
		return;
	}
	if (tp==1){
		for(int i=0; i<n; ++i)
			d[i] *= out[i]*(1-out[i]);
		return;
	}
	assert(false);
}

inline void hznn2::wdata::init_rand(){
	w = ( rand() % 2001 - 1000 ) / 5000.0;
	s = 0;
}

inline void hznn2::wdata::grad(double dx){
	s += dx * dx;
	w -= dx / sqrt(s + 0.5) * 0.05;
}

void hznn2::mkmat(wdata **&w, wdata *&c, int l_n, int r_n){
	w = new wdata*[l_n];
	w[0] = new wdata[l_n * r_n];
	for(int i=1; i<l_n; ++i)
		w[i] = w[i-1] + r_n;
	c = new wdata[r_n];

	for(int i=0; i<l_n; ++i)
	for(int j=0; j<r_n; ++j)
		w[i][j].init_rand();

	for(int i=0; i<r_n; ++i)
		c[i].init_rand();
}

void hznn2::MAT::init(LAY *_l, LAY *_r){
	assert( l = _l );
	assert( r = _r );
	l_n = l->n;
	r_n = r->n;
	lo = l->out;
	ri = r->in;
	sta = new int[r_n];

	mkmat(w,c,l_n,r_n);

	if (l_n<300 || r_n<300){
		tp=0;
		return;
	}

	tp = 1;
	mid_n = 10;
	mkmat(w1,c1,l_n,mid_n);
	mkmat(w2,c2,mid_n,r_n);
	mid = new double[mid_n];
	mid_d = new double[mid_n];
	pd = new double[r_n];
}

void hznn2::localc(wdata **w, wdata *c, int l_n, int r_n, double *lo, double *ri){
	for(int i=0; i<r_n; ++i) ri[i] = c[i].w;

	for(int i=0; i<l_n; ++i) if (lo[i]>0)
	for(int j=0; j<r_n; ++j)
		ri[j] += lo[i] * w[i][j].w;
}

int hznn2::MAT::calc(){
	if (tp==0){
		localc(w,c,l_n,r_n,lo,ri);
		return 0;
	}

	// tp==1
	localc(w1,c1,l_n,mid_n,lo,mid);
	localc(w2,c2,mid_n,r_n,mid,pd);

	int top=0;

	for(int i=0; i<r_n; ++i)
	if (pd[i]>0){
		sta[top++] = i;
		ri[i] = c[i].w;
	} else ri[i] = 0;

	for(int i=0; i<l_n; ++i) if (lo[i]>0)
	for(int k=0; k<top; ++k)
		ri[sta[k]] += lo[i] * w[i][sta[k]].w;

	for(int i=0; i<r_n; ++i){
		pd[i] -= ri[i];
		c2[i].grad(pd[i]);
	}

	for(int i=0; i<mid_n; ++i){
		mid_d[i] = 0;
		for(int j=0; j<r_n; ++j){
			mid_d[i] += pd[j] * w2[i][j].w;
			w2[i][j].grad(mid[i] * pd[j]);
		}
		c1[i].grad(mid_d[i]);
	}

	for(int i=0; i<l_n; ++i)
	for(int j=0; j<mid_n; ++j)
		w1[i][j].grad(lo[i] * mid_d[j]);

	return top;
}

void hznn2::MAT::bp(){
	int top = 0;
	double *rd = r->d;
	double *ld = l->d;

	for(int i=0; i<r_n; ++i)
	if (rd[i]){
		sta[top++] = i;
		c[i].grad(rd[i]);
	}

	for(int i=0; i<l_n; ++i)
		ld[i]=0;

	for(int i=0; i<l_n; ++i)
	if (lo[i] > 0){
		for(int k=0; k<top; ++k){
			int j=sta[k];
			ld[i] += rd[j] * w[i][j].w;
			w[i][j].grad(lo[i]*rd[j]);
		}
	}
}

void hznn2::init(int _lay_num, const int _n[]){
	lay_num = _lay_num;
	for(int i=0; i<lay_num; ++i)
		lay[i].init(_n[i], i==lay_num-1);

	for(int i=0; i<lay_num-1; ++i)
		mat[i].init(&lay[i],&lay[i+1]);
}

std::pair<int,int> hznn2::calc(const double f[]){
	memcpy(lay[0].out, f, sizeof(double)*lay[0].n);
	int s2 = 0;
	for(int i=1; i<lay_num; ++i){
		s2 += mat[i-1].calc();
		lay[i].calc_out();
	}
	int s = 0;
	for(int i=0; i<lay[1].n; ++i) s += (lay[1].out[i]>0);
	return std::make_pair(s,s2);
}

void hznn2::get_output(double f[]){
	memcpy(f, lay[lay_num-1].out, sizeof(double)*lay[lay_num-1].n);
}

void hznn2::bp(const double f[]){
	for(int i=0; i<lay[lay_num-1].n; ++i)
		lay[lay_num-1].d[i] = lay[lay_num-1].out[i] - f[i];

	for(int i=lay_num-1; i>0; --i){
		lay[i].bp_d();
		mat[i-1].bp();
	}
};
