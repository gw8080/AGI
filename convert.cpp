#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;
vector<string> getFiles(string file)
{
    vector<string> res;
    ifstream myfileV (file);
    string word, line, total;
    if (myfileV.is_open())
    {
        while ( getline (myfileV,line) )
        {
            res.push_back(line);
        }
        myfileV.close();
    }
    return res;
}
int main()
{

    vector<string> files = getFiles("images.txt");
    cout << "Processing";
    for(int a = 0; a < files.size(); a++)
    {
        string command = "magick " + files[a] + " " + files[a];
        system(command.c_str());
        cout << ".";
    }

    return 0;
}
