using System;
using System.Diagnostics;

namespace NeuralNetwork
{
    internal class Program
    {
        static void Main()
        {
            //int[] NetStructure = new int[] { 784, 50, 10 };
            NeuralNetwork network = IOReader.LoadNetwork("path_to_saved_network_file");//new NeuralNetwork(NetStructure);  

            double[][] TrainData = IOReader.GetTrainingDataInputs();
            double[][] TrainLabels = IOReader.GetTrainingDataOutputs();
            double[][] TestData = IOReader.GetTestDataInputs();
            double[][] TestLabels = IOReader.GetTestDataOutputs();


            Algorithm.TrainNetwork(network, TrainData, TrainLabels, TestData, TestLabels, 0.02);
        }


        static void Benchmark(int iterations)
        {
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            for (int i = 0; i < iterations; i++)
            {
                //test function 1 goes here
            }
            stopwatch.Stop();
            Console.WriteLine($"Function 1 speed time test results: {stopwatch.Elapsed}.");
            stopwatch.Reset();

            stopwatch.Start();
            for (int i = 0; i < iterations; i++)
            {
                //test function 2 goes here
            }
            stopwatch.Stop();
            Console.WriteLine($"Function 2 speed time test results: {stopwatch.Elapsed}.");
            stopwatch.Reset();
        }
    }
}
