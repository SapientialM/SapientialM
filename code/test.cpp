#include<algorithm>
#include<vector>
#include<iostream>
#include<string>
#include<unordered_map>

using namespace std;

bool isIsomorphic(string s, string t);
int main(int argc, char **argv){
    string s = "aaa";
    string t = "BBB";
    cout << isIsomorphic(s, t) << endl;
};

bool isIsomorphic(string s, string t) {
    unordered_map<char, char>::iterator iterS;
    unordered_map<char, char>::iterator iterT;
    unordered_map<char, char> mapS;
    unordered_map<char, char> mapT;
    for(int i = 0; i < s.size(); i ++){
        iterS = mapS.find(s[i]);
        cout<<"Finding "<<s[i]<<endl;
        if(iterS != mapS.end()){
            cout<<"Find the "<<s[i]<<endl;
            iterT = mapT.find(t[i]);
            if(iterT == mapT.end()){
                return false;
            }
            cout<<"Find the "<<t[i]<<endl;
            if(!(iterS->second == t[i] && iterT->second == s[i])){
                cout<<"not equal "<<iterT->second<<"!="<<iterS->second<<endl;
                return false;
            }
        }
        else{
            iterT = mapT.find(t[i]);
            if(iterT != mapT.end()){
                return false;
            }
            mapS[s[i]] = t[i];
            mapT[t[i]] = s[i];
        }
    }
    return true;
}