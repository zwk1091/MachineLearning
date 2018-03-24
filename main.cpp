#include<iostream>
#include<vector>
#include<string>
#include<fstream>
		
using namespace std;
int main() {
	const int NUMOFSAMPLE=5;
	const int NUMOFSUBJECT=5;
	const double LEARNINGRATE = 0.00001;
	vector<double> parameter(NUMOFSUBJECT,0);
	vector<double> preParameter(NUMOFSUBJECT);

	//input data
	ifstream in("Text.txt");
	int temp;
	vector<vector<int>> data(NUMOFSAMPLE,vector<int>(NUMOFSUBJECT));
	for (int i = 0; i <NUMOFSUBJECT; i++) {
		for (int j = 0; j < NUMOFSAMPLE; j++) {
			in >> temp;
			data[j][i]=temp;
		}
	}
	while (1) {
		cout << "按任意键进行迭代" << endl;
		string t;
		cin >> t;
		//show the parameter
		cout << "the parameter: ";
		for (int i = 0; i < NUMOFSUBJECT; i++) {
			cout << parameter[i] << " ";
		}
		//show the cost function

		double Cost;
		double sumInCost = 0;
		for (int i = 0; i < NUMOFSAMPLE; i++) {
			//compute h(x)
			int hx = parameter[0];
			for (int j = 0; j < NUMOFSUBJECT; j++) {
				hx += parameter[j] * data[i][j];
			}
			sumInCost += (hx - data[i][0])*(hx - data[i][0]);
		}
		Cost = (1 / (2 * (double)NUMOFSAMPLE))*sumInCost;
		cout << "cost function: " << Cost << endl;

		preParameter = parameter;
		for (int i = 0; i < NUMOFSUBJECT; i++) {
			//compute sum
			double sum = 0;

			for (int j = 0; j < NUMOFSAMPLE; j++) {
				//compute h(x)
				int hx = preParameter[0];
				for (int k = 1; k < NUMOFSUBJECT; k++) {
					hx += preParameter[k] * data[j][k];
				}
				sum += (hx - data[j][0])*data[j][i];
			}

			parameter[i] = preParameter[i] - LEARNINGRATE*(1 / (double)NUMOFSAMPLE)*sum;
		}
		
	}
	system("pause");
	return 0;

}