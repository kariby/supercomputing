#include <iostream>
#include <math.h>
#include <time.h>
using namespace std;

int main()
{
	clock_t start, end;
	int n=15;
	float S;
	S = 0;
	cin >> n;
	start = clock();
	for (int i = 1; i <= n; ++i) {
		S = S + (i%2 ? -1 : 1 )*log(i) / double(i);
	}
	cout << "S = " << S << "\n";
	end = clock();
	printf("The above code block was executed in %.4f second(s)\n", ((double)end - start) / ((double)CLOCKS_PER_SEC));
	system("pause");
	return 0;
}