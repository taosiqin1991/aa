#include <vector>
#include <string>

using namespace std;

// 304  O(mn)

class Solution{
public:
    int longestIncreasingPath(vector<vector<int>>& mat){
        if(mat.size()==0 || mat[0].size()==0 ) return 0;

        M = mat.size();
        N = mat.size();
        auto mem = vector<vector<int>>(M, vector<int>(N, 0));

        int ans =0;
        for(int i=0; i<M; i++){
            for(int j=0; j<N; j++){
                int tmp = dfs(mat, i, j, mem);
                ans = max(ans, tmp);
            }
        }
        return ans;

    }

    int dfs(vector<vector<int>>& mat, int i, int j, vector<vector<int>>& mem){
        if( mem[i][j] !=0)  return mem[i][j];

        mem[i][j]++;
        for(int k=0; k<4; k++){
            int ki = i + dirs[k][0];
            int kj = j + dirs[k][1];

            if(ki>=0 && ki<M && kj>=0 && kj<N){

            }
        }
    }


private:
    static constexpr int dirs[4][2] = {{-1,0}, {1,0}, {0,-1},{0,1}};
    int M;
    int N;




}