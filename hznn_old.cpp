#include <bits/stdc++.h>
using namespace std;

//#define double float

struct HZNN {
	static const int N=10000, LAY=8;

	struct wdata {
		double w,s;

		inline void init_rand(){
			w = ( rand() % 2001 - 1000 ) / 5000.0;
			s = 0;
		}

		inline void init_0(){
			w = s = 0;
		}

		inline void grad(double dx){
			s += dx * dx;
			w -= dx / sqrt(s + 0.5) * 0.05;
		}
	};

	int n,sz[LAY],st[LAY+1],lay;
	wdata *w[N],c[N];

	void init(int _lay, const int _sz[]){
		lay = _lay;
		for(int i=0; i<lay; ++i){
			sz[i] = _sz[i];
			st[i+1] = st[i] + sz[i];
		}
		n = st[lay];

		for(int i=0; i<lay-1; ++i){

			wdata *p = new wdata [sz[i]*sz[i+1]];

			for(int j=0; j<sz[i]; ++j){
				w[st[i]+j] = p + j*sz[i+1];
			}

			for(int j=st[i]; j<st[i+1]; ++j)
			for(int k=0; k<sz[i+1]; ++k){
				w[j][k].init_rand();
			}
		}

		for(int i=0; i<n; ++i){
			c[i].init_0();
		}
	}

	void write_data(const char name[]){
		FILE *f = fopen(name,"wb");
		fwrite(&n,sizeof(int),1,f);
		fwrite(&lay,sizeof(int),1,f);
		fwrite(sz,sizeof(int)*lay,1,f);
		fwrite(st,sizeof(int)*(lay+1),1,f);

		for(int i=0; i<lay-1; ++i){
			fwrite(w[st[i]],sizeof(wdata)*sz[i]*sz[i+1],1,f);
		}

		fwrite(c,sizeof(wdata)*n,1,f);

		fclose(f);
	}

	void read_data(const char name[]){
		FILE *f = fopen(name,"rb");
		fread(&n,sizeof(int),1,f);
		fread(&lay,sizeof(int),1,f);
		fread(sz,sizeof(int)*lay,1,f);
		fread(st,sizeof(int)*(lay+1),1,f);

		for(int i=0; i<lay-1; ++i){
			wdata *p = new wdata [sz[i]*sz[i+1]];
			fread(p,sizeof(wdata)*sz[i]*sz[i+1],1,f);

			for(int j=0; j<sz[i]; ++j){
				w[st[i]+j] = p + j*sz[i+1];
			}
		}

		fread(c,sizeof(wdata)*n,1,f);

		fclose(f);
	}


	double fin[N],fout[N],d[N];

	void calc(const double f[]){
		memcpy(fin,f,sizeof(double)*sz[0]);

		for(int i=0; i<lay-1; ++i){

			for(int j=st[i]; j<st[i+1]; ++j){
				fout[j] = fin[j]>0 ? fin[j] : 0;
			}

			for(int k=0; k<sz[i+1]; ++k){
				fin[st[i+1]+k] = c[st[i+1]+k].w;
			}

			for(int j=st[i]; j<st[i+1]; ++j)
			if (fin[j]>0){
				for(int k=0; k<sz[i+1]; ++k){
					fin[st[i+1]+k] += fout[j] * w[j][k].w;
				}
			}
		}

		for(int i=st[lay-1]; i<n; ++i){
			fout[i] = 1.0 / ( 1.0 + exp(-fin[i]) );
		}
	}

	void get_output(double f[]){
		memcpy(f,fout+st[lay-1],sizeof(double)*sz[lay-1]);
	}

	int s1[N],s2[N],t1,t2;
	double mx = 0;

	void back(const double f[]){
		for(int i=0; i<sz[lay-1]; ++i){
			d[st[lay-1]+i] = fout[st[lay-1]+i] - f[i];
		}

		for(int i=lay-1; i>0; --i){

			if (i==lay-1){
				for(int j=st[i]; j<st[i+1]; ++j){
					d[j] *= fout[j] * (1-fout[j]);
				}
			} else {
				for(int j=st[i]; j<st[i+1]; ++j){
					if (fin[j]<0) d[j]=0;
				}
			}

			for(int j=st[i]; j<st[i+1]; ++j){
				c[j].grad(d[j]);
				mx=max(mx,c[j].s);
			}

			for(int j=st[i-1]; j<st[i]; ++j) d[j]=0;

			t1=t2=0;
			for(int j=st[i-1]; j<st[i]; ++j) if (fin[j]>0) s1[t1++]=j;
			for(int j=st[i]; j<st[i+1]; ++j) if (i==lay-1 || fin[j]>0) s2[t2++]=j;

			//printf("%lf\n",double(t1+t2)/double(sz[i-1]+sz[i]));


			for(int i1=0; i1<t1; ++i1)
			//for(int j=st[i-1]; j<st[i]; ++j)
			for(int i2=0; i2<t2; ++i2){
			//for(int k=st[i]; k<st[i+1]; ++k){
				int j = s1[i1];
				int k = s2[i2];
				d[j] += d[k] * w[j][k-st[i]].w;
				w[j][k-st[i]].grad(fout[j]*d[k]);
				mx=max(mx,w[j][k-st[i]].s);
			}


			/*
			for(int j=st[i-1]; j<st[i]; ++j)
			for(int k=st[i]; k<st[i+1]; ++k){
				d[j] += d[k] * w[j][k-st[i]].w;
				w[j][k-st[i]].grad(fout[j]*d[k]);
			}
			*/
		}
	}
};

