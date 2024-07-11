using System;
namespace NeuralNetworks
{
    internal class Program
    {
        static void Main()
        {
            LayerNetwork network = IOReader.LoadNetworkFromPath(@"...\Utils\ConvergenceTestnet.csv"); //the preceding path must be modified to reflect your own project directory 

            double[][] TrainData = IOReader.GetTrainingDataInputs();
            double[][] TrainLabels = IOReader.GetTrainingDataOutputs();
            double[][] TestData = IOReader.GetTestDataInputs();
            double[][] TestLabels = IOReader.GetTestDataOutputs();

            StochasticGradDescent Algorithm = new StochasticGradDescent();
            Algorithm.BenchmarkConvergence(network, TrainData, TrainLabels, TestData, TestLabels);
            Console.ReadLine();
        }
    }
}
