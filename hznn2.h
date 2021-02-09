
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

	struct MAT {
		LAY *l, *r;
		int l_n, r_n;
		double *lo, *ri;
		wdata **w, *c;
		int *sta;

		void init(LAY *_l, LAY *_r);
		void calc();
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
	int calc(const double f[]);
	void get_output(double f[]);
	void bp(const double f[]);
};
