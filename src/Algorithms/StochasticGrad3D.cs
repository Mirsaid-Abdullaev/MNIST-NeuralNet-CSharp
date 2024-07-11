using System;
using System.Diagnostics;
using static NeuralNetworks.MathUtil;

/*
 * This class was created as the training algorithm class for the OptimisedNetwork class, also inheriting the IAlgorithm interface
 * to maintain OOP best practices and keep the code easily maintainable and scalable. As talked about in the README, this version 
 * used 3D arrays instead of keeping a Layer class, which 3x the time taken to train per epoch so this Network class is not recommended
 * to be used because of its poor performance compared to using the Layer class.
*/


namespace NeuralNetworks
{

    internal class StochasticGrad3D: IAlgorithm
    {
        public override void TrainNetwork(INetwork TrainingNetwork, double[][] TrainData, double[][] TrainLabels, double[][] TestData, double[][] TestLabels, double LearnRate, int Epochs)
        {
            try
            {
                var temp = (OptimisedNetwork)TrainingNetwork;
            }
            catch
            {
                throw new Exception("Error: current network instance is not of type LayerNetwork.");
            }
            this.Network = (OptimisedNetwork)TrainingNetwork;

            Stopwatch stopwatch = new Stopwatch();
            float prev_accuracy = GetPercentageAccuracy(TestData, TestLabels);
            Console.WriteLine($"Epoch 0: initialisation. Initial performance: {prev_accuracy}.");            
            for (int epoch = 1; epoch < Epochs; epoch++)
            {
                stopwatch.Start();
                for (int i = 0; i < TrainData.Length; i++)
                {
                    ForwardProp(TrainData[i]);
                    CalculateGradients(TrainLabels[i]);
                    UpdateParameters(LearnRate);
                }
                stopwatch.Stop();
                TimeSpan temp = stopwatch.Elapsed;
                stopwatch.Restart();
                float accuracy = GetPercentageAccuracy(TestData, TestLabels);
                if (accuracy > prev_accuracy) //need to save network
                {
                    Network.SaveNetwork("TEST1");
                }
                prev_accuracy = accuracy;
                stopwatch.Stop();
                Console.WriteLine($"Epoch {epoch} completed. Current performance %: {accuracy}. Epoch training time: {temp}. Test run time: {stopwatch.Elapsed}");
                stopwatch.Reset();
            }
        }

        protected override double[] ForwardProp(double[] Inputs)
        {
            /*
                * 1) set first layer outputs as inputs to the forward prop - must be 1d array
                * 2) for each pair of curr and prev layer, calculate the necessary outputs of the next layers
                * 3) return resulting array of outputs
            */
            var CurrNet = (OptimisedNetwork)Network;
            CurrNet.Outputs[0] = Inputs; //step 1

            for (int LayerIndex = 1; LayerIndex < CurrNet.LayerCount; LayerIndex++)
            {
                for (int currNeuron = 0; currNeuron < CurrNet.NeuronCounts[LayerIndex]; currNeuron++)
                {
                    double output = CurrNet.Biases[LayerIndex - 1][currNeuron];
                    for (int prevNeuron = 0; prevNeuron < CurrNet.NeuronCounts[LayerIndex - 1]; prevNeuron++)
                    {
                        output += CurrNet.Outputs[LayerIndex - 1][prevNeuron] * CurrNet.Weights[LayerIndex - 1][prevNeuron][currNeuron];
                    }
                    CurrNet.Outputs[LayerIndex][currNeuron] = Sigmoid(output);
                }
            }
            return CurrNet.Outputs[^1];
        }

        protected override int ForwardPropClassify(double[] Inputs)
        {
            /*
             * 1) set first layer outputs as inputs to the forward prop - must be 1d array
             * 2) for each pair of curr and prev layer, calculate the necessary outputs of the next layers
             * 3) return the index of the maximum value in the final output array (index of classification i.e. label)
            */
            var CurrNet = (OptimisedNetwork)Network;
            CurrNet.Outputs[0] = Inputs;

            for (int LayerIndex = 1; LayerIndex < CurrNet.LayerCount; LayerIndex++)
            {
                for (int currNeuron = 0; currNeuron < CurrNet.NeuronCounts[LayerIndex]; currNeuron++)
                {
                    double output = CurrNet.Biases[LayerIndex - 1][currNeuron];
                    for (int prevNeuron = 0; prevNeuron < CurrNet.NeuronCounts[LayerIndex - 1]; prevNeuron++)
                    {
                        output += CurrNet.Outputs[LayerIndex - 1][prevNeuron] * CurrNet.Weights[LayerIndex - 1][prevNeuron][currNeuron];
                    }
                    CurrNet.Outputs[LayerIndex][currNeuron] = Sigmoid(output);
                }
            }

            //series of comparisons to get the index of the max value from the final layer outputs
            double max = -1;
            int index = -1;
            double[] Outputs = CurrNet.Outputs[^1];
            if (Outputs[0] > max)
            {
                index = 0;
                max = Outputs[0];
            }
            if (Outputs[1] > max)
            {
                index = 1;
                max = Outputs[1];
            }
            if (Outputs[2] > max)
            {
                index = 2;
                max = Outputs[2];
            }
            if (Outputs[3] > max)
            {
                index = 3;
                max = Outputs[3];
            }
            if (Outputs[4] > max)
            {
                index = 4;
                max = Outputs[4];
            }
            if (Outputs[5] > max)
            {
                index = 5;
                max = Outputs[5];
            }
            if (Outputs[6] > max)
            {
                index = 6;
                max = Outputs[6];
            }
            if (Outputs[7] > max)
            {
                index = 7;
                max = Outputs[7];
            }
            if (Outputs[8] > max)
            {
                index = 8;
                max = Outputs[8];
            }
            if (Outputs[9] > max)
            {
                index = 9;
            }

            return index;
        }

        protected override void CalculateGradients(double[] Target)
        {
            /*
             * 1) calculate last layer dCdZ and save it to the array
             * 2) for each pair of curr and next layers, calculate the previous dC/dZ (backpropagation)
             * FORMULAS:
             * 1) dC/dz(n) = 2(f(z(n)) - y) f'(z), where a(n) = f(z(n)) = final outputs
             * 2) dC/dA(n-1) = dC/dz(n) * w(n)
             * 3) dC/dZ(n-1) = dC/dA(n-1) * f'(z(n-1))
            */
            var CurrNet = (OptimisedNetwork)Network;
            for (int i = 0; i < CurrNet.NeuronCounts[^1]; i++) //setting the dC/dZ(n) values of the last layer neurons using FORMULA 1
            {
                CurrNet.dCdz[^1][i] = 2 * (CurrNet.Outputs[^1][i] - Target[i]) * Sigmoid_Deriv(CurrNet.Outputs[^1][i]);
            }


            for (int LayerIndex = CurrNet.LayerCount - 2; LayerIndex > 0; LayerIndex--)
            {
                int NeuronCount = CurrNet.NeuronCounts[LayerIndex];
                int NextNeuronCount = CurrNet.NeuronCounts[LayerIndex + 1];
                for (int currNeuron = 0; currNeuron < NeuronCount; currNeuron++)
                {
                    double sum_dCdA = 0;
                    for (int nextNeuron = 0; nextNeuron < NextNeuronCount; nextNeuron++)
                    {
                        sum_dCdA += CurrNet.Weights[LayerIndex - 1][currNeuron][nextNeuron] * CurrNet.dCdz[LayerIndex + 1][nextNeuron]; //working out dC/da(n-1) using FORMULA 2
                    }
                    CurrNet.dCdz[LayerIndex][currNeuron] = sum_dCdA * Sigmoid_Deriv(CurrNet.Outputs[LayerIndex][currNeuron]); //working out dC/dZ(n-1) using FORMULA 3
                }
            }
            Network = CurrNet;
        }
        protected override void UpdateParameters(double LearnRate)
        {
            /*
             * 1) for each pair of curr and prev layers, use curr dC/dZ(n) and prev a(n-1) to find dC/dW(n)
             * 2) use dC/dW(n) * learnrate and subtract from current w(n)
             * 3) use dC/db(n) = dC/dZ(n), to subtract dCdB(n) * learnrate from b(n)
             * 
             * FORMULAS:
             * 4) dC/dw(n) = dC/dz(n) * a(n-1), n >= 1
             * 5) dC/dB(n) = dC/dz(n)
            */
            var CurrNet = (OptimisedNetwork)Network;
            for (int LayerIndex = 1; LayerIndex < CurrNet.LayerCount; LayerIndex++)
            {
                int CurrNeuronCount = CurrNet.NeuronCounts[LayerIndex];
                int PrevNeuronCount = CurrNet.NeuronCounts[LayerIndex - 1];
                for (int currNeuron = 0; currNeuron < CurrNeuronCount; currNeuron++)
                {
                    for (int prevNeuron = 0; prevNeuron < PrevNeuronCount; prevNeuron++)
                    {
                        CurrNet.Weights[LayerIndex - 1][prevNeuron][currNeuron] -= LearnRate * CurrNet.dCdz[LayerIndex][currNeuron] * CurrNet.Outputs[LayerIndex - 1][prevNeuron];  //updating weight between prev and curr neuron using dC/dw(n) by FORMULA 4
                        CurrNet.Biases[LayerIndex - 1][currNeuron] -= LearnRate * CurrNet.dCdz[LayerIndex][currNeuron]; //same as above but using FORMULA 5, so not multiplying by a(n - 1) as dz(n)/dB(n) = 1
                    }
                }
            }
            Network = CurrNet;
        }

        public override float GetPercentageAccuracy(double[][] TestData, double[][] TestLabels)
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
        public override void BenchmarkConvergence(INetwork Network, double[][] TrainData, double[][] TrainLabels, double[][] TestData, double[][] TestLabels)
        {
            try
            {
                var temp = (OptimisedNetwork)Network;
            }
            catch
            {
                throw new Exception("Error: current network instance is not of type LayerNetwork.");
            }
            this.Network = (OptimisedNetwork)Network;

            Stopwatch stopwatch = new Stopwatch();
            float accuracy = GetPercentageAccuracy(TestData, TestLabels);
            TimeSpan total_training = new TimeSpan(0);
            TimeSpan total_validating = new TimeSpan(0);
            float initial_accuracy = accuracy;
            Console.WriteLine($"Epoch 0: initialisation. Initial performance: {accuracy}.");
            for (int epoch = 1; epoch < 100; epoch++)
            {
                stopwatch.Start();
                for (int i = 0; i < TrainData.Length; i++)
                {
                    ForwardProp(TrainData[i]);
                    CalculateGradients(TrainLabels[i]);
                    UpdateParameters(0.0001);
                }
                stopwatch.Stop();
                TimeSpan temp = stopwatch.Elapsed;
                total_training += temp;
                stopwatch.Restart();
                accuracy = GetPercentageAccuracy(TestData, TestLabels);
                stopwatch.Stop();
                total_validating += stopwatch.Elapsed;
                Console.WriteLine($"Epoch {epoch} completed. Current performance %: {accuracy}. Epoch training time: {temp}. Test run time: {stopwatch.Elapsed}");
                stopwatch.Reset();
            }
            Console.WriteLine();
            Console.WriteLine("-------Benchmark complete. Results:-------");
            Console.WriteLine($"Final accuracy rate (after 100 epochs): {accuracy}%");
            Console.WriteLine($"Average accuracy improvement per epoch: {(accuracy - initial_accuracy) / 100}%");
            Console.WriteLine($"Average time to run a training epoch: {total_training / 100}");
            Console.WriteLine($"Average time to run a validation epoch: {total_validating / 100}%");
        }
    }
}
