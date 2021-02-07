
struct hznn {
	int in, mid, out, pre;

	struct wdata {
		double w,s;
		inline void init_rand();
		inline void grad(double dx);
	};

	static const int N=5000;

	wdata *w[N], c1[N], c2[N];

	void init(int _in, int _mid, int _out, int _pre);
	//void write_data(const char file_name[]);
	//void read_data(const char file_name[]);

	double in1[N], in2[N], fout[N], d[N];
	int sta[N], top;

	int calc(const double f[]);
	void get_output(double f[]);
	void bp(const double f[]);
};
