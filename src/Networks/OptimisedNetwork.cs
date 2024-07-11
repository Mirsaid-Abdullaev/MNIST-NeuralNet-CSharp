using System;
using System.IO;
using System.Text;

namespace NeuralNetworks
{
    internal class OptimisedNetwork: INetwork
    {
        public double[][][] Weights; //first dimension is layer number, second dimension is previous neuron index, third is current neuron index
        public double[][] Biases; //first dimension is layer number, second is the current neuron index
        public double[][] Outputs; //first dimension is layer number, second is current neuron index - holds activated outputs
        public double[][] dCdz; //first dimension is layer number, second is current neuron index

        public int[] NeuronCounts; //holds the number of neurons in each layer;
        public int LayerCount;

        public OptimisedNetwork(int[] LayerStructure)
        {
            NeuronCounts = LayerStructure;
            LayerCount = LayerStructure.Length;

            Weights = new double[LayerCount - 1][][]; //there are weight matrices between each pair of layers 
            Biases = new double[LayerCount - 1][];
            Outputs = new double[LayerCount][];
            dCdz = new double[LayerCount][];

            for (int i = 0; i < LayerCount - 1; i++)
            {
                int prevNeurons = NeuronCounts[i];
                int currNeurons = NeuronCounts[i + 1];
                Weights[i] = new double[prevNeurons][];
                Biases[i] = new double[currNeurons];
                Outputs[i] = new double[prevNeurons];
                dCdz[i] = new double[prevNeurons];

                double temp = Math.Sqrt(NeuronCounts[i]); //sqrt(prevneuroncount)
                for (int j = 0; j < prevNeurons; j++) //this loop sets up the weights matrix
                {
                    Weights[i][j] = new double[currNeurons];
                    for (int k = 0; k < currNeurons; k++)
                    {
                        Weights[i][j][k] = GetRandom() / temp;
                    }
                }
                for (int j = 0; j < currNeurons; j++) //this loop sets up the bias matrix
                {
                    Biases[i][j] = GetRandom() / temp;
                }
            }

            Outputs[^1] = new double[NeuronCounts[^1]]; //the last layer still needs an output
            dCdz[^1] = new double[NeuronCounts[^1]]; //same reason as above
        }
        public OptimisedNetwork(double[][][] Weights, double[][] Biases, int[] NeuronCounts, int LayerCount)
        {
            this.Weights = Weights;
            this.Biases = Biases;
            this.NeuronCounts = NeuronCounts;
            this.LayerCount = LayerCount;

            Outputs = new double[LayerCount][];
            dCdz = new double[LayerCount][];
            for (int i = 0; i < LayerCount; i++)
            {
                Outputs[i] = new double[NeuronCounts[i]];
                dCdz[i] = new double[NeuronCounts[i]];
            }
        }

        static double GetRandom() //y = sqrt(-2 * ln(1 - x1)) * cos(2 * pi * x2), 0 < x1 <= 1, 0 < x2 <= 1
        {
            Random random = new Random();
            return Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble());
        }

        public override void SaveNetwork(string FileName)
        {
            string FilePath = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData) + @"\MNIST_Net\" + FileName + ".csv";
            if (!Directory.Exists(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData) + @"\MNIST_Net\"))
            {
                Directory.CreateDirectory(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData) + @"\MNIST_Net\");
            }
            using StreamWriter sw = new StreamWriter(FilePath, false);
            sw.Write(GetNetworkData());
            sw.Flush();
        }
        protected override string GetNetworkData()
        {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.AppendLine(LayerCount.ToString());
            stringBuilder.AppendLine(NeuronCounts[0].ToString() + ",0,");

            for (int i = 1; i < LayerCount; i++)
            {
                stringBuilder.Append(NeuronCounts[i].ToString() + "," + NeuronCounts[i - 1] + ",");
                for (int j = 0; j < NeuronCounts[i - 1]; j++)
                {
                    stringBuilder.Append(string.Join(",", Weights[i - 1][j]) + ",");
                }
                stringBuilder.AppendLine(string.Join(",", Biases[i - 1]));
            }
            return stringBuilder.ToString();
        }
    }
}
