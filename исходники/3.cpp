#include <mpi.h>
#include <iostream>
#include <math.h>
#include <cstdio>

using namespace std;
const long long n = 1400000000;

int main(int argc, char *argv[])
{
	int i, rank, size, namelen;
	char name[MPI_MAX_PROCESSOR_NAME];
	double sum = 0;
	MPI_Init(&argc, &argv);
	MPI_Status stat;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double startt = MPI_Wtime();
	int m, start, width;
	m = n%size;
	width = n / size;
	if (rank<m)
	{
		width++;
		start = rank*(n / size + 1);
	}
	else
	{
		start = m*(n / size + 1) + (rank - m)*n / size;
	}
	for (long long i = start + 2; i < start + width + 2; i++)
		sum += (i % 2 ? -1 : 1)*log(i) / double(i);
	printf("%lf\n", sum);
	if (rank == 0)
	{
		double s;
		for (int i = 1; i < size; i++)
		{
			MPI_Recv(&s, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &stat);
			sum += s;
		}


	}
	else
	{
		MPI_Send(&sum, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
	}
	double end = MPI_Wtime();
	double search_time = end - startt;
	cout << "sum = " << sum << endl;
	cout << "time = " << search_time << endl;

	MPI_Finalize();

}
