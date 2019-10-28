

bool isPossible(int row, int col, int N, vector<bool>& b_Column,
	vector<bool>& b_MainDiagonal, vector<bool>& b_CounerDiagonal);
void search(int row, vector<int>& path, int N, vector<vector<int> >&all, vector<bool>& b_Column,
	vector<bool>& b_MainDiagonal, vector<bool>& b_CounerDiagonal);
int Queen(int N, vector<vector<int> >& all);
void Print(vector<vector<int> >& all);


// N: number of the queen
// all: all the available deployment
// return the total number
int Queen(int N, vector<vector<int> >& all) {
	vector<int> path(N, -1);
	vector<bool> b_Column(N, false);        // cannot meet in column
	vector<bool> b_MainDiagonal(2 * N - 1, false);    // cannot meet in main diagnoal
	vector<bool> b_CounerDiagonal(2 * N - 1, false);  // cannot meet in couter diagnonal

	// recursive and hidden constrain: not to meet in row
	search(0, path, N, all, b_Column, b_MainDiagonal, b_CounerDiagonal);  

	return all.size();
}

void search(int row, vector<int>& path, int N, vector<vector<int> >&all, vector<bool>& b_Column,
	vector<bool>& b_MainDiagonal, vector<bool>& b_CounerDiagonal) {
	if (row == N ) {
		all.push_back(path);
		return;
	}
	for (int col = 0; col < N; col++) {
		if (isPossible(row, col, N, b_Column, b_MainDiagonal, b_CounerDiagonal)) {
			path[row] = col;
			b_Column[col] = true;
			b_MainDiagonal[row - col + N - 1] = true;
			b_CounerDiagonal[row + col] = true;
			search(row + 1, path, N, all, b_Column, b_MainDiagonal, b_CounerDiagonal);

			// backtracking
			b_Column[col] = false;
			b_MainDiagonal[row - col + N - 1] = false;
			b_CounerDiagonal[row + col] = false;
		}
	}
}

bool isPossible(int row, int col, int N, vector<bool>& b_Column,
	vector<bool>& b_MainDiagonal, vector<bool>& b_CounerDiagonal) {
	if (b_Column[col])
		return false;
	if (b_MainDiagonal[row - col + N - 1])
		return false;
	if (b_CounerDiagonal[row + col])
		return false;
	return true;
}

void Print(vector<vector<int> >& all) {
	int M = all.size();
	int N;
	if (M > 0)
		N = all[0].size();
	else
		N = 0;
	cout << "There are " << M << " results" << endl;
	for (int i = 0; i < M; i++) {
		cout << endl;
		for (int j = 0; j < N; j++) {
			cout << all[i][j] << "  ";
		}
		cout << endl;
	}
	cout << "There are " << M << " results" << endl;
}