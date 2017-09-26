#include<bits/stdc++.h>
#include<random>
#define LEARNING_RATE 0.3
#define MOMENTUM 0.3
using namespace std;

mt19937 rng;
vector<map<int,double> > weights;
vector<map<int,double> > previousdel;
vector<vector<int> > network;
int totalneuroncount = 0;
double error;

void addDenseLayer(int neuronnumber,int flagoutput)
{	
	int lastlayer = network.size() - 1;
	vector<int> newlayer;
	for(int i = 1;i <= neuronnumber;i++)
	{
		newlayer.push_back(totalneuroncount++);
		map<int, double> newweights;
		weights.push_back(newweights);
		previousdel.push_back(newweights);
	}
	network.push_back(newlayer); //adding new layer to the existing network
	if(lastlayer < 0) //When addding first layer, declaration of new weights is not required.
		return;
	int newlayernumber = network.size() - 1;
	rng.seed(random_device()());
    uniform_real_distribution<double> udouble_dist(-0.05,0.05);
	for(int i = 0;i < network[lastlayer].size(); i++) //declare new weights after adding new layer
	{
		for(int j=flagoutput?0:1;j < neuronnumber;j++)
		{
			weights[network[lastlayer][i]][network[newlayernumber][j]] = udouble_dist(rng); 
			previousdel[network[lastlayer][i]][network[newlayernumber][j]] = 0;
		}
	}
}

void forwardpropagation(vector<double> inputs,vector<double> &neuronoutput)
{
		for(int i=0;i<network.size();i++)
		{
			if(i==0) //first layer 
			{
				for(register int j=0;j<network[i].size();j++)
					neuronoutput[network[i][j]] = inputs[j];
			}
			else
			{
				for(register int j=0;j<network[i].size();j++)
				{
					if(j==0 && i!=(network.size()-1)) //bias unit
						neuronoutput[network[i][j]] = 1;
					else
					{
						double weightedinput = 0.0;
						for(register int k=0;k<network[i-1].size();k++)
						{
							weightedinput += (weights[network[i-1][k]][network[i][j]] * neuronoutput[network[i-1][k]]);
						}
						neuronoutput[network[i][j]] = 1.0/(1.0+exp(-weightedinput));
					}
				}
			}
		}
}

void backpropagation(vector<map<int,double> > &olddelta,vector<double> &neuronoutput,vector<double> &outputs)
{
	vector<double> delta(totalneuroncount,0);
		//propagating the errors backwards through the network
		for(int i=network.size()-1;i>=0;i--)
		{
			if(i==network.size()-1) //calculating errors for the output layer
			{
				for(register int j=0;j<network[i].size();j++)
				{
					double thisneuronoutput = neuronoutput[network[i][j]];
					delta[network[i][j]] = thisneuronoutput * (1.0 - thisneuronoutput) * (outputs[j] - thisneuronoutput);
					error = abs(error + delta[network[i][j]]);
				}
			}
			else //calculating errors for the hidden layers
			{
				for(register int j=0;j<network[i].size();j++)
				{
					double thisneuronoutput = neuronoutput[network[i][j]];
					for(register int k=0;k<network[i+1].size();k++)
					{
						delta[network[i][j]] += (weights[network[i][j]][network[i+1][k]] * delta[network[i+1][k]]);
					}
					delta[network[i][j]] *= (thisneuronoutput*(1.0-thisneuronoutput));
					error = abs(error + delta[network[i][j]]);
				}
			}
		}
		//update the weights
		for(int i=0;i<weights.size();i++)
		{
			for(map<int,double>::iterator it=weights[i].begin();it!=weights[i].end();++it)
			{
				double newdelta = (LEARNING_RATE * delta[it->first] * neuronoutput[i]) + (MOMENTUM * olddelta[i][it->first]);
				weights[i][it->first] += newdelta;
				olddelta[i][it->first] = newdelta;
			}
		}
	
}

void train(string filename,vector<map<int,double> > &olddelta)
{
	//reading the file contents and populating the input pixels in the pixels matrix. Also flattens the pixels matrix to input to the neural net
	vector<double> inputs((30*32)+1, -1);	
	int inputcount = 0;
	inputs[inputcount++] = 1; //bias unit
	string filelocation = "faces_4/" + filename;
	FILE *f = fopen(filelocation.c_str(),"rb");
	int pixels[30][32] = {0};
	int m,n,maxgreyscaleval;
	char pgmtype[3];
	fgets(pgmtype,3,f);
	fscanf(f,"%d %d %d",&m,&n,&maxgreyscaleval);
	fgetc(f);
	for(int i=0;i<n;i++)
	{
		int c;
		for(int j=0;j<m;j++)
		{
			c = fgetc(f);
			pixels[i][j] = c;
			inputs[inputcount++] = (double)pixels[i][j]/255.0; //scaling down the inputs to range from 0 to 1
		}
	}
	//determining the pose of the person in the image - straight,left,right or up
	vector<double> outputs(20,0.1);
	string an2i = "an2i";
	string at33 = "at33";
	string boland = "boland";
	string bpm = "bpm";
	string ch4f = "ch4f";
	string cheyer = "cheyer";
	string choon = "choon";
	string danieln = "danieln";
	string glickman = "glickman";
	string karyadi = "karyadi";
	string kawamura = "kawamura";
	string kk49 = "kk49";
	string megak = "megak";
	string mitchell = "mitchell";
	string night = "night";
	string phoebe = "phoebe";
	string saavik = "saavik";
	string steffi = "steffi";
	string sz24 = "sz24";
	string tammo = "tammo";
	if(filename.find(an2i)!=string::npos)
		outputs[0]=0.9;
	else if(filename.find(at33)!=string::npos)
		outputs[1]=0.9;
	else if(filename.find(boland)!=string::npos)
		outputs[2]=0.9;
	else if(filename.find(bpm)!=string::npos)
		outputs[3]=0.9;
	else if(filename.find(ch4f)!=string::npos)
		outputs[4]=0.9;
	else if(filename.find(cheyer)!=string::npos)
		outputs[5]=0.9;
	else if(filename.find(choon)!=string::npos)
		outputs[6]=0.9;
	else if(filename.find(danieln)!=string::npos)
		outputs[7]=0.9;
	else if(filename.find(glickman)!=string::npos)
		outputs[8]=0.9;
	else if(filename.find(karyadi)!=string::npos)
		outputs[9]=0.9;
	else if(filename.find(kawamura)!=string::npos)
		outputs[10]=0.9;
	else if(filename.find(kk49)!=string::npos)
		outputs[11]=0.9;
	else if(filename.find(megak)!=string::npos)
		outputs[12]=0.9;
	else if(filename.find(mitchell)!=string::npos)
		outputs[13]=0.9;
	else if(filename.find(night)!=string::npos)
		outputs[14]=0.9;
	else if(filename.find(phoebe)!=string::npos)
		outputs[15]=0.9;
	else if(filename.find(saavik)!=string::npos)
		outputs[16]=0.9;
	else if(filename.find(steffi)!=string::npos)
		outputs[17]=0.9;
	else if(filename.find(sz24)!=string::npos)
		outputs[18]=0.9;
	else if(filename.find(tammo)!=string::npos)
		outputs[19]=0.9;
	else
		cout<<"Invalid filename\n";
	vector<double> neuronoutput(totalneuroncount);
	forwardpropagation(inputs,neuronoutput);
	backpropagation(olddelta,neuronoutput,outputs);
	fclose(f);
}

double findaccuracy(char *testfile)
{
	int correct,total;
	correct = total = 0;
	fstream g(testfile,ios::in|ios::out);
	string filename;
	g>>filename;	
	while(!g.eof())	
	{
		//reading the file contents and populating the input pixels in the pixels matrix. Also flattens the pixels matrix to input to the neural net
		vector<double> inputs((30*32)+1, -1);	
		int inputcount = 0;
		inputs[inputcount++] = 1; //bias unit
		string filelocation = "faces_4/" + filename;
		FILE *f = fopen(filelocation.c_str(),"rb");
		int pixels[30][32] = {0};
		int m,n,maxgreyscaleval;
		char pgmtype[3];
		fgets(pgmtype,3,f);
		fscanf(f,"%d %d %d",&m,&n,&maxgreyscaleval);
		fgetc(f);
		for(int i=0;i<n;i++)
		{
			int c;
			for(int j=0;j<m;j++)
			{
				c = fgetc(f);
				pixels[i][j] = c;
				inputs[inputcount++] = (double)pixels[i][j]/255.0; //scaling down the inputs to range from 0 to 1
			}
		}
		//determining the person in the image
		int face = -1;
		string an2i = "an2i";
		string at33 = "at33";
		string boland = "boland";
		string bpm = "bpm";
		string ch4f = "ch4f";
		string cheyer = "cheyer";
		string choon = "choon";
		string danieln = "danieln";
		string glickman = "glickman";
		string karyadi = "karyadi";
		string kawamura = "kawamura";
		string kk49 = "kk49";
		string megak = "megak";
		string mitchell = "mitchell";
		string night = "night";
		string phoebe = "phoebe";
		string saavik = "saavik";
		string steffi = "steffi";
		string sz24 = "sz24";
		string tammo = "tammo";
		if(filename.find(an2i)!=string::npos)
			face=0;
		else if(filename.find(at33)!=string::npos)
			face=1;
		else if(filename.find(boland)!=string::npos)
			face=2;
		else if(filename.find(bpm)!=string::npos)
			face=3;
		else if(filename.find(ch4f)!=string::npos)
			face=4;
		else if(filename.find(cheyer)!=string::npos)
			face=5;
		else if(filename.find(choon)!=string::npos)
			face=6;
		else if(filename.find(danieln)!=string::npos)
			face=7;
		else if(filename.find(glickman)!=string::npos)
			face=8;
		else if(filename.find(karyadi)!=string::npos)
			face=9;
		else if(filename.find(kawamura)!=string::npos)
			face=10;
		else if(filename.find(kk49)!=string::npos)
			face=11;
		else if(filename.find(megak)!=string::npos)
			face=12;
		else if(filename.find(mitchell)!=string::npos)
			face=13;
		else if(filename.find(night)!=string::npos)
			face=14;
		else if(filename.find(phoebe)!=string::npos)
			face=15;
		else if(filename.find(saavik)!=string::npos)
			face=16;
		else if(filename.find(steffi)!=string::npos)
			face=17;
		else if(filename.find(sz24)!=string::npos)
			face=18;
		else if(filename.find(tammo)!=string::npos)
			face=19;
		else
			cout<<"Invalid filename\n";
		vector<double> neuronoutput(totalneuroncount);
		forwardpropagation(inputs,neuronoutput);
		double maxprob = 0.0;
		int networkoutput;
		for(int i=0;i<network[network.size()-1].size();i++)
		{
			if(neuronoutput[network[network.size()-1][i]] > maxprob)
			{
				maxprob = neuronoutput[network[network.size()-1][i]];
				networkoutput = i;
			}
		}
		if(networkoutput == face)
			correct ++;
		else
			cout<<"The recognizer misclassified "<<filename<<endl;
		total++;
		g>>filename;
		fclose(f);
	}
	g.close();
	return (double)correct/(double)total;
}

int main(int argc, char **argv)
{
	if(argc < 3)
	{
		cout<<"Please specify the training and testing datasets"<<endl;
		return 0;
	}
	//Create the network architecture
	addDenseLayer((30*32)+1,0); //First layer which takes the inputs plus a bias unit
	addDenseLayer(20+1,0); //Adding second layer with 6 hidden units plus a bias unit
	addDenseLayer(20,1); //Adding final layer with only one neuron
	//Output encoding: 4 output nodes for 4 poses. The output is the highest of the values output by the nodes
	fstream f(argv[1],ios::in|ios::out);
	vector<string> filenames;
	string filename;
	f>>filename;	
	while(!f.eof())	
	{
		f>>filename;
		filenames.push_back(filename);
	}
	for(int i=1;i<=27700;i++)
	{

		rng.seed(random_device()());
    	uniform_int_distribution<int> uint_dist(0,276);
    	int randindex = uint_dist(rng);
		train(filenames[randindex],previousdel);
			
		if(i%277==0)
		{
			cout<<"Iteration "<<(i/277)<<endl;
			double testaccuracy = findaccuracy(argv[2]);
			cout<<"Accuracy on testing dataset after iteration "<<(i/277)<<" is "<<testaccuracy<<endl;
		}
	}
	cout<<"The final accuracy on the testing dataset is "<<findaccuracy(argv[2])<<endl;

	return 0;
}
