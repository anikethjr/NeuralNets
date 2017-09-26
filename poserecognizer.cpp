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
		for(int i=0;i<network[network.size()-1].size();i++)
		{
			if(neuronoutput[network[network.size()-1][i]] > maxprob)
			{
				maxprob = neuronoutput[network[network.size()-1][i]];
				networkoutput = i;
			}
		}
		if(networkoutput == pose)
			correct ++;
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
	addDenseLayer(6+1,0); //Adding second layer with 6 hidden units plus a bias unit
	addDenseLayer(4,1); //Adding final layer with only one neuron
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
			double testaccuracy = findaccuracy(argv[2]);
			cout<<"Accuracy on testing dataset after iteration "<<(i/277)<<" is "<<testaccuracy<<endl;
		}
	}
	cout<<"The final accuracy on the testing dataset is "<<findaccuracy(argv[2])<<endl;

	return 0;
}
