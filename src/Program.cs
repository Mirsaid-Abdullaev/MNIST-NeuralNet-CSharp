namespace NeuralNetwork
{
    internal class Program
    {
        static void Main()
        {
            NeuralNetwork network = IOReader.LoadNetworkFromPath(@"...\Utils\ConvergenceTestnet.csv"); //the preceding path must be modified to reflect your own project directory 

            double[][] TrainData = IOReader.GetTrainingDataInputs();
            double[][] TrainLabels = IOReader.GetTrainingDataOutputs();
            double[][] TestData = IOReader.GetTestDataInputs();
            double[][] TestLabels = IOReader.GetTestDataOutputs();

            Algorithm.BenchmarkConvergence(network, TrainData, TrainLabels, TestData, TestLabels);
        }
    }
}
