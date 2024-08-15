namespace NeuralNetworks
{
    internal abstract class IMBGDAlgorithm //for mini-batch-based algorithms
    {
        public INetwork Network;
        protected abstract int ForwardPropClassify(double[] Inputs);
        protected abstract void ForwardProp(double[][] Inputs);
        protected abstract void CalculateGradients(double[][] Target);
        protected abstract void UpdateParameters(double LEARN_RATE);
        public abstract void TrainNetwork(INetwork Network, double[][] TrainData, double[][] TrainLabels, double[][] TestData, double[][] TestLabels, double LEARN_RATE, int Epochs);
        public float GetPercentageAccuracy(double[][] TestData, double[][] TestLabels)
        {
            int count = 0;
            double max = -1;
            int index = -1;
            for (int i = 0; i < TestData.Length; i++)
            {
                int Label = ForwardPropClassify(TestData[i]);

                if (TestLabels[i][0] > max)
                {
                    index = 0;
                    max = TestLabels[i][0];
                }
                if (TestLabels[i][1] > max)
                {
                    index = 1;
                    max = TestLabels[i][1];
                }
                if (TestLabels[i][2] > max)
                {
                    index = 2;
                    max = TestLabels[i][2];
                }
                if (TestLabels[i][3] > max)
                {
                    index = 3;
                    max = TestLabels[i][3];
                }
                if (TestLabels[i][4] > max)
                {
                    index = 4;
                    max = TestLabels[i][4];
                }
                if (TestLabels[i][5] > max)
                {
                    index = 5;
                    max = TestLabels[i][5];
                }
                if (TestLabels[i][6] > max)
                {
                    index = 6;
                    max = TestLabels[i][6];
                }
                if (TestLabels[i][7] > max)
                {
                    index = 7;
                    max = TestLabels[i][7];
                }
                if (TestLabels[i][8] > max)
                {
                    index = 8;
                    max = TestLabels[i][8];
                }
                if (TestLabels[i][9] > max)
                {
                    index = 9;
                }
                if (Label == index) //checking if the maximum value is in the same index for expected and actual output
                {
                    count++;
                }
                max = -1;
            }
            return (float)count / TestData.Length * 100;
        }
        public abstract void BenchmarkConvergence(INetwork Network, double[][] TrainData, double[][] TrainLabels, double[][] TestData, double[][] TestLabels);
    }
}
