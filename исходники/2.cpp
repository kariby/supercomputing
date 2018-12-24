#include<iostream>
#include"math.h"
#include<ctime>
#include"omp.h"

using namespace std;

int main()
{
	long n = 1400000000;
	double sum = 0;
	double start, end, time;

	start = omp_get_wtime();

#pragma omp parallel for reduction (+:sum)
	for (int i = 1; i <= n, i++)
		sum += pow(i, 1.0 / 3.0) / ((i + 1.0)*sqrt(i));

	end = omp_get_wtime();

	time = end - start;

	cout << "Result" << sum << endl;
	cout << "Time" << time << endl;
	system("pause");
	return 0;
}
