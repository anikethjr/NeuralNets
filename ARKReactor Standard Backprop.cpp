#include<bits/stdc++.h>
#include<random>
#define LEARNING_RATE 0.3
#define MOMENTUM 0.3
using namespace std;

mt19937 rng;
vector<map<int,double> > weights;
vector<map<int,double> > previousdel;
vector<map<int,double> > delta;
vector<vector<int> > network;
int totalneuroncount = 0;
double error;

void addDenseLayer(int neuronnumber)
{	
	int lastlayer = network.size() - 1;
	vector<int> newlayer;
	for(int i = 1;i <= neuronnumber;i++)
	{
		newlayer.push_back(totalneuroncount++);
		map<int, double> newweights;
		weights.push_back(newweights);
		previousdel.push_back(newweights);
		delta.push_back(newweights);
	}
	network.push_back(newlayer); //adding new layer to the existing network
	if(lastlayer < 0) //When addding first layer, declaration of new weights is not required.
		return;
	int newlayernumber = network.size() - 1;
	rng.seed(random_device()());
    uniform_real_distribution<double> udouble_dist(0,0.05);
	for(int i = 0;i < network[lastlayer].size(); i++) //declare new weights after adding new layer
	{
		for(int j=0;j < neuronnumber;j++)
		{
			weights[network[lastlayer][i]][network[newlayernumber][j]] = lastlayer==0?0:udouble_dist(rng); 
			previousdel[network[lastlayer][i]][network[newlayernumber][j]] = 0;
			delta[network[lastlayer][i]][network[newlayernumber][j]] = 0;
			cout<<"The weight for the edge from "<<network[lastlayer][i]<<" to "<<network[newlayernumber][j]<<" is initialized to "<<weights[network[lastlayer][i]][network[newlayernumber][j]]<<endl;
		}
	}
}

int binarytoint(string bin)
{
	int size = bin.size();
	int intval = 0;
	for(int i=size-1;i>=0;i--)
	{
		intval = intval + pow(2,size-1-i)*(bin[i]-'0');
	}
	return intval;
}

void forwardpropagation(vector<double> inputs,vector<double> &neuronoutput)
{
	#pragma omp parallel for
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

void backpropagation(vector<double> &neuronoutput,vector<double> &outputs)
{
	vector<double> del(totalneuroncount,0);
	#pragma omp parallel for
		//propagating the errors backwards through the network
		for(int i=network.size()-1;i>=0;i--)
		{
			if(i==network.size()-1) //calculating errors for the output layer
			{
				for(register int j=0;j<network[i].size();j++)
				{
					double thisneuronoutput = neuronoutput[network[i][j]];
					del[network[i][j]] = thisneuronoutput * (1.0 - thisneuronoutput) * (outputs[j] - thisneuronoutput);
					error = abs(error + del[network[i][j]]);
				}
			}
			else //calculating errors for the hidden layers
			{
				for(register int j=0;j<network[i].size();j++)
				{
					double tempdelta = 0.0;
					double thisneuronoutput = neuronoutput[network[i][j]];
					for(register int k=0;k<network[i+1].size();k++)
					{
						tempdelta += (weights[network[i][j]][network[i+1][k]] * del[network[i+1][k]]);
					}
					tempdelta *= (thisneuronoutput*(1.0-thisneuronoutput));
					del[network[i][j]] = tempdelta;
					error = abs(error + del[network[i][j]]);
				}
			}
		}
		//update deltas
		for(int i=0;i<weights.size();i++)
		{
			for(map<int,double>::iterator it=weights[i].begin();it!=weights[i].end();++it)
			{
				delta[i][it->first] += (LEARNING_RATE * del[it->first] * neuronoutput[i] * weights[i][it->first]);
			}
		}
}

void updateweights()
{
	//update the weights
	for(int i=0;i<weights.size();i++)
	{
		for(map<int,double>::iterator it=weights[i].begin();it!=weights[i].end();++it)
		{			
			weights[i][it->first] += delta[i][it->first] + (MOMENTUM * previousdel[i][it->first]);
			previousdel[i][it->first] = delta[i][it->first];
			delta[i][it->first] = 0;
		}
	}
}

void train(string filename)
{
	//reading the file contents and populating the input pixels in the pixels matrix. Also flattens the pixels matrix to input to the neural net
	vector<double> inputs(32*30, -1);
	int inputcount = 0;
	string filelocation = "faces_4/" + filename;
	FILE *f = fopen(filelocation.c_str(),"rb");
	int pixels[32][30] = {0};
	int m,n,maxgreyscaleval;
	char pgmtype[3];
	fscanf(f,"%s",pgmtype);
	fscanf(f,"%d %d\n%d",&m,&n,&maxgreyscaleval);
	for(int i=0;i<32;i++)
	{
		char c;
		for(int j=0;j<30;j++)
		{
			c = fgetc(f);
			pixels[i][j] = (unsigned)c;
			inputs[inputcount++] = (double)c/255.0; //scaling down the inputs to range from 0 to 1
		}
	}

	//determining the pose of the person in the image - straight,left,right or up
	vector<double> outputs(4,0);
	string straight = "straight";
	string left = "left";
	string right = "right";
	string up = "up";
	//the output value is set to 0.9 for the desired output, others are set to 0.1
	if(filename.find(straight)!=string::npos)
	{
		outputs[0]=0.9;
		outputs[1]=0.1;
		outputs[2]=0.1;
		outputs[3]=0.1;
	}	
	else if(filename.find(left)!=string::npos)
	{
		outputs[0]=0.1;
		outputs[1]=0.9;
		outputs[2]=0.1;
		outputs[3]=0.1;
	}
	else if(filename.find(right)!=string::npos)
	{
		outputs[0]=0.1;
		outputs[1]=0.1;
		outputs[2]=0.9;
		outputs[3]=0.1;
	}
	else if(filename.find(up)!=string::npos)
	{
		outputs[0]=0.1;
		outputs[1]=0.1;
		outputs[2]=0.1;
		outputs[3]=0.9;
	}
	else
		cout<<"Invalid filename\n";
	vector<double> neuronoutput(totalneuroncount);
	forwardpropagation(inputs,neuronoutput);
	backpropagation(neuronoutput,outputs);
	fclose(f);
}

int findaccuracy()
{
	int correct,total;
	correct = total = 0;
	fstream g("all_train.list.txt",ios::in|ios::out);
	string filename;
	g>>filename;	
	while(!g.eof())	
	{
		//reading the file contents and populating the input pixels in the pixels matrix. Also flattens the pixels matrix to input to the neural net
		vector<double> inputs(32*30, -1);
		int inputcount = 0;
		string filelocation = "faces_4/" + filename;
		FILE *f = fopen(filelocation.c_str(),"rb");
		int pixels[32][30] = {0};
		int m,n,maxgreyscaleval;
		char pgmtype[3];
		fscanf(f,"%s",pgmtype);
		fscanf(f,"%d %d\n%d",&m,&n,&maxgreyscaleval);
		for(int i=0;i<32;i++)
		{
			char c;
			for(int j=0;j<30;j++)
			{
				c = fgetc(f);
				pixels[i][j] = c;
				inputs[inputcount++] = (double)c/255.0; //scaling down the inputs to range from 0 to 1
			}
		}

		//determining the pose of the person in the image - straight,left,right or up
		vector<double> outputs(4,0);
		int pose = -1;
		string straight = "straight";
		string left = "left";
		string right = "right";
		string up = "up";
		if(filename.find(straight)!=string::npos)
		{
			pose=0;
		}	
		else if(filename.find(left)!=string::npos)
		{
			pose=1;
		}
		else if(filename.find(right)!=string::npos)
		{
			pose=2;
		}
		else if(filename.find(up)!=string::npos)
		{
			pose=3;
		}
		else
			cout<<"Invalid filename\n";
		vector<double> neuronoutput(totalneuroncount);
		forwardpropagation(inputs,neuronoutput);
		double maxprob = 0.0;
		int networkoutput;
		//cout<<filename<<" ";
		for(int i=0;i<network[network.size()-1].size();i++)
		{
			//cout<<neuronoutput[network[network.size()-1][i]]<<" ";
			if(neuronoutput[network[network.size()-1][i]] > maxprob)
			{
				maxprob = neuronoutput[network[network.size()-1][i]];
				networkoutput = i;
			}
		}
		//cout<<networkoutput<<" ";
		if(networkoutput == pose)
			correct ++;
		total++;
		g>>filename;
		fclose(f);
		//cout<<endl;
	}
	g.close();

	cout<<"Accuracy on dataset is "<<(double)correct/(double)total<<endl;
	return (double)correct/(double)total;
}

int main()
{
	//Create the network architecture
	addDenseLayer(32*30); //First layer which takes the inputs
	addDenseLayer(6); //Adding second layer with 6 hidden units
	addDenseLayer(4); //Adding final layer with only one neuron
	//Output encoding: 4 output nodes for 4 poses. The output is the highest of the values output by the nodes
	fstream f("all_train.list.txt",ios::in|ios::out);
	for(int i=0;i<10000;i++)
	{
		error = 0;
		string filename;
		f>>filename;	
		//cout<<filename<<endl;
		while(!f.eof())	
		{
			train(filename);
			f>>filename;
			
			//cout<<filename<<endl;
		}
		updateweights();
		double testaccuracy = findaccuracy();
		cout<<error<<endl;
		f.clear();
		f.seekg(0, ios::beg);
		
	}
	
	return 0;
}
