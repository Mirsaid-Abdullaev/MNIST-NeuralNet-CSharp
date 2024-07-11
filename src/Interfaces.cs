namespace NeuralNetworks
{
    internal abstract class INetwork
    {
        public abstract void SaveNetwork(string FileName);
        protected abstract string GetNetworkData();
    }
    internal abstract class IAlgorithm
    {
        public INetwork Network;
        protected abstract int ForwardPropClassify(double[] Inputs);
        protected abstract double[] ForwardProp(double[] Inputs);
        protected abstract void CalculateGradients(double[] Target);
        protected abstract void UpdateParameters(double LearnRate);
        public abstract void TrainNetwork(INetwork Network, double[][] TrainData, double[][] TrainLabels, double[][] TestData, double[][] TestLabels, double LearnRate, int Epochs);
        public abstract float GetPercentageAccuracy(double[][] TestData, double[][] TestLabels);
        public abstract void BenchmarkConvergence(INetwork Network, double[][] TrainData, double[][] TrainLabels, double[][] TestData, double[][] TestLabels);
    }


}
