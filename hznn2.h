
struct hznn2 {

	struct LAY {
		int n, tp;
		// tp==0 : RELU
		// tp==1 : sigmoid
		double *in, *out, *d;
		void init(int _n, int _tp);
		void calc_out();
		void bp_d();
	};

	struct wdata {
		double w,s;
		inline void init_rand();
		inline void grad(double dx);
	};

	static void mkmat(wdata **&w, wdata *&c, int l_n, int r_n);

	static void localc(wdata **w, wdata *c, int l_n, int r_n, double *lo, double *ri);

	struct MAT {
		LAY *l, *r;
		int l_n, r_n;
		double *lo, *ri;
		wdata **w, *c;
		int *sta;
		int tp;
		// tp==0 : old
		// tp==1 : optimize
		wdata **w1, *c1;
		wdata **w2, *c2;
		double *mid, *mid_d, *pd;
		int mid_n;

		void init(LAY *_l, LAY *_r);
		int calc();
		void bp();
	};

	static const int N=20;
	LAY lay[N];
	int lay_num;

	MAT mat[N];
	/*
	int mat_num;

	int calc_mat[N][N];
	int bp_mat[N][N];
	*/

	void init(int _lay_num, const int _n[]);
	std::pair<int,int> calc(const double f[]);
	void get_output(double f[]);
	void bp(const double f[]);
};
