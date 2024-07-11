using System;
using System.Diagnostics;
using static NeuralNetworks.MathUtil;

namespace NeuralNetworks
{
    internal class StochasticGradDescent: IAlgorithm
    {
        protected override int ForwardPropClassify(double[] Inputs)
        {
            /*
             * 1) set first layer outputs as inputs to the forward prop - must be 1d array
             * 2) for each pair of curr and prev layer, calculate the necessary outputs of the next layers
             * 3) return the index of the maximum value in the final output array (index of classification i.e. label)
            */
            var CurrNet = (LayerNetwork)Network;

            for (int i = 0; i < Inputs.Length; i++)
            {
                CurrNet.Layers[0].Outputs[i] = Inputs[i];
            }


            for (int i = 1; i < CurrNet.Layers.Length; i++) //forward prop
            {
                Layer currLayer = CurrNet.Layers[i];
                Layer prevLayer = CurrNet.Layers[i - 1];
                for (int currNeuron = 0; currNeuron < currLayer.NeuronCount; currNeuron++)
                {
                    double output = currLayer.Bias[currNeuron];
                    for (int prevNeuron = 0; prevNeuron < prevLayer.NeuronCount; prevNeuron++) //for each neuron in the previous letter
                    {
                        output += prevLayer.Outputs[prevNeuron] * currLayer.Weights[prevNeuron][currNeuron];
                    }
                    currLayer.Outputs[currNeuron] = Sigmoid(output);
                }
            }

            //series of comparisons to get the index of the max value from the final layer outputs
            double max = -1;
            int index = -1;
            double[] Outputs = CurrNet.Layers[^1].Outputs;
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
        protected override double[] ForwardProp(double[] Inputs)
        {
            /*
             * 1) set first layer outputs as inputs to the forward prop - must be 1d array
             * 2) for each pair of curr and prev layer, calculate the necessary outputs of the next layers
             * 3) return resulting array of outputs
            */
            var CurrNet = (LayerNetwork)Network;

            for (int i = 0; i < Inputs.Length; i++)
            {
                CurrNet.Layers[0].Outputs[i] = Inputs[i];
            }
            int NumLayers = CurrNet.Layers.Length;
            for (int i = 1; i < NumLayers; i++) //forward prop
            {
                Layer currLayer = CurrNet.Layers[i];
                Layer prevLayer = CurrNet.Layers[i - 1];
                for (int currNeuron = 0; currNeuron < currLayer.NeuronCount; currNeuron++)
                {
                    double output = currLayer.Bias[currNeuron];
                    for (int prevNeuron = 0; prevNeuron < prevLayer.NeuronCount; prevNeuron++) //for each neuron in the previous letter
                    {
                        output += prevLayer.Outputs[prevNeuron] * currLayer.Weights[prevNeuron][currNeuron];
                    }
                    currLayer.Outputs[currNeuron] = Sigmoid(output);
                }
            }
            return CurrNet.Layers[^1].Outputs;
        }

        //  ---- BACKPROPAGATION ---- //
        // IMPORTANT!
        // Cost function C is referenced all throughout this backprop section, any cost functions can be used, but respective changes to the formulas must be made due to calculus chain rule differentiation
        // In this case it is: C = (a(n) - y)^2, or the Mean Squared Error (MSE) function, where a(n) = actual outputs, and y = expected outputs.
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
            var CurrNet = (LayerNetwork)Network;

            for (int i = 0; i < CurrNet.Layers[^1].NeuronCount; i++) //setting the dC/dZ(n) values of the last layer neurons using FORMULA 1
            {
                CurrNet.Layers[^1].dCdZ[i] = 2 * (CurrNet.Layers[^1].Outputs[i] - Target[i]) * Sigmoid_Deriv(CurrNet.Layers[^1].Outputs[i]); 
            }

            for (int LIndex = CurrNet.Layers.Length - 2; LIndex > 0; LIndex--)
            {
                Layer currLayer = CurrNet.Layers[LIndex]; //layer n - 1
                Layer nextLayer = CurrNet.Layers[LIndex + 1]; //layer n

                for (int currNeuron = 0; currNeuron < currLayer.NeuronCount; currNeuron++)
                {
                    double Sum_dCdA = 0;
                    for (int nextNeuron = 0; nextNeuron < nextLayer.NeuronCount; nextNeuron++)
                    {
                        Sum_dCdA += nextLayer.Weights[currNeuron][nextNeuron] * nextLayer.dCdZ[nextNeuron]; //working out dC/da(n-1) using FORMULA 2
                    }
                    currLayer.dCdZ[currNeuron] = Sum_dCdA * Sigmoid_Deriv(currLayer.Outputs[currNeuron]); //working out dC/dZ(n-1) using FORMULA 3
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
             * 6) new_w(x) = old_w(x) - learnrate * dC/dW(x)
             * 7) new_b(x) = old_b(x) - learnrate * dC/db(x)
            */
            var CurrNet = (LayerNetwork)Network;

            for (int LIndex = 1; LIndex < CurrNet.Layers.Length; LIndex++)
            {
                Layer currLayer = CurrNet.Layers[LIndex];
                Layer prevLayer = CurrNet.Layers[LIndex - 1];
                for (int currNeuron = 0; currNeuron < currLayer.NeuronCount; currNeuron++)
                {
                    for (int prevNeuron = 0; prevNeuron < prevLayer.NeuronCount; prevNeuron++)
                    {
                        double dCdW = currLayer.dCdZ[currNeuron] * prevLayer.Outputs[prevNeuron]; //dC/dW calculation using FORMULA 4
                        double dCdB = currLayer.dCdZ[currNeuron]; //dCdB using FORMULA 5;
                        currLayer.Weights[prevNeuron][currNeuron] -= LearnRate * dCdW; //updating weight between prev and curr neuron using dC/dw(n) by FORMULA 4 and by FORMULA 6
                        currLayer.Bias[currNeuron] -= LearnRate * dCdB; //same as above but using FORMULA 5 and FORMULA 7, so not multiplying by a(n - 1) as dz(n)/dB(n) = 1
                    }
                }
            }
            Network = CurrNet;
        }
        // --- END BACKPROPAGATION --- //

        public override void TrainNetwork(INetwork TrainingNetwork, double[][] TrainData, double[][] TrainLabels, double[][] TestData, double[][] TestLabels, double LearnRate, int Epochs)
        {
            try
            {
                var temp = (LayerNetwork) TrainingNetwork;
            }
            catch
            {
                throw new Exception("Error: current network instance is not of type LayerNetwork.");
            }
            this.Network = (LayerNetwork)TrainingNetwork;

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
                    TrainingNetwork.SaveNetwork("TEST1");
                }
                prev_accuracy = accuracy;
                stopwatch.Stop();
                Console.WriteLine($"Epoch {epoch} completed. Current performance %: {accuracy}. Epoch training time: {temp}. Test run time: {stopwatch.Elapsed}");
                stopwatch.Reset();
            }
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
        public override void BenchmarkConvergence(INetwork LayerNetwork, double[][] TrainData, double[][] TrainLabels, double[][] TestData, double[][] TestLabels)
        {
            try
            {
                var temp = (LayerNetwork)LayerNetwork;
            }
            catch
            {
                throw new Exception("Error: current network instance is not of type LayerNetwork.");
            }
            this.Network = (LayerNetwork)LayerNetwork;

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
