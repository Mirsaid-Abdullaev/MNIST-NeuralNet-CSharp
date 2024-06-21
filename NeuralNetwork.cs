using System;
using System.IO;
using System.Runtime.Serialization;
using System.Text;

namespace NeuralNetwork
{
    internal class NeuralNetwork
    {
        public Layer[] Layers;
        public NeuralNetwork(int[] LayersData) //LayersData holds number of neurons in each layer in the network e.g. {2, 3, 1}, {100, 150, 50, 10, 8} etc.
        {
            Layers = new Layer[LayersData.Length];
            Layers[0] = new Layer(LayersData[0], 0);
            for (int i = 1; i < Layers.Length; i++)
            {
                Layers[i] = new Layer(LayersData[i], LayersData[i - 1]);
            }
        }

        public NeuralNetwork(Layer[] Layers) //for loading from storage
        {
            this.Layers = Layers;
        }

        public void SaveNetwork(string FileName)
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

        private string GetNetworkData()
        {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.AppendLine(Layers.Length.ToString());
            for (int i = 0;i < Layers.Length - 1; i++)
            {
                stringBuilder.AppendLine(Layers[i].GetLayerData());
            }
            stringBuilder.Append(Layers[^1].GetLayerData());
            return stringBuilder.ToString();
        }
    }
    internal class Layer
    {
        public int NeuronCount; //number of neurons in the current layer
        public int PrevNeuronCount; //number of neurons in prev layer

        public readonly double[] Bias; //Each index relates to each neuron in this layer (b(n))
        public readonly double[] Outputs; //As above, but each index is the same neuron's output value (a(n) = f(z(n)))
        public readonly double[] dCdZ; //As above, but each index is the same neuron's dC/dZ value (dC/dZ(n) = dC/da(n) * da(n)/dz(n) = 2 * (a(n) - y) * f'(z(n)))
        public readonly double[][] Weights;
        public Layer(int Neurons, int PrevNeurons) //setting all weights and biases for all neurons in one go
        {
            NeuronCount = Neurons;
            PrevNeuronCount = PrevNeurons;
            if (PrevNeuronCount > 0) //checking not input layer
            {
                double temp = Math.Sqrt(PrevNeuronCount);
                Weights = new double[PrevNeuronCount][];
                Bias = new double[NeuronCount];
                dCdZ = new double[NeuronCount];
                for (int i = 0; i < PrevNeuronCount; i++)
                {
                    Weights[i] = new double[NeuronCount];
                    for (int j = 0; j < NeuronCount; j++)
                    {
                        Weights[i][j] = GetRandom() / temp;
                    }
                }
                for (int i = 0; i < NeuronCount; i++)
                {
                    Bias[i] = GetRandom() / temp;
                }
            }
            Outputs = new double[NeuronCount]; //all layers have output layer, as input layer has a(0) which are just the network inputs
        }
        public Layer(int Neurons, int PrevNeurons, double[][] Weights, double[] Bias) //for loading a layer from storage
        {
            NeuronCount = Neurons;
            PrevNeuronCount = PrevNeurons;
            if(PrevNeuronCount > 0) //input layer
            {
                this.Weights = Weights;
                this.Bias = Bias;
                dCdZ = new double[NeuronCount];
            }
            Outputs = new double[NeuronCount];
        }

        static double GetRandom() //y = sqrt(-2 * ln(1 - x1)) * cos(2 * pi * x2), 0 < x1 <= 1, 0 < x2 <= 1
        {
            Random random = new Random();
            return Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble());
        }

        public string GetLayerData()
        {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.Append(NeuronCount.ToString() + "," + PrevNeuronCount.ToString() + ",");
            if (PrevNeuronCount != 0)
            {
                for (int i = 0; i < PrevNeuronCount; i++)
                {
                    stringBuilder.Append(string.Join(",", Weights[i]) + ",");
                }
                stringBuilder.Append(string.Join(",", Bias));
            }
            return stringBuilder.ToString();
        }
    }

    internal class OptimisedNetwork
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

        public void SaveNetwork(string FileName)
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
        private string GetNetworkData()
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
