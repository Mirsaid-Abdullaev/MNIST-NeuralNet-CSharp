using System;
using static NeuralNetworks.MathUtil;
namespace NeuralNetworks
{
    internal class Program
    {
        static void Main()
        {
            double[][] TrainData = IOReader.GetTrainingDataInputs();
            double[][] TrainLabels = IOReader.GetTrainingDataOutputs();
            double[][] TestData = IOReader.GetTestDataInputs();
            double[][] TestLabels = IOReader.GetTestDataOutputs();

            SGDMomentum Algorithm = new SGDMomentum();

            MomentumNetwork network = IOReader.LoadMTMNetworkFromPath(@"path\of\project\folder\network_csv_file.csv"); //the preceding path must be modified to reflect your own project directory 
            Algorithm.BenchmarkConvergence(network, TrainData, TrainLabels, TestData, TestLabels);
            Console.ReadLine();
        }
    }
}
