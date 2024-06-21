using System;
using System.Diagnostics;

namespace NeuralNetwork
{
    internal static class Algorithm
    {
        private static int ForwardPropClassify(Layer[] Layers, double[] Inputs)
        {
            /*
             * 1) set first layer outputs as inputs to the forward prop - must be 1d array
             * 2) for each pair of curr and prev layer, calculate the necessary outputs of the next layers
             * 3) return the index of the maximum value in the final output array (index of classification i.e. label)
            */
            for (int i = 0; i < Inputs.Length; i++)
            {
                Layers[0].Outputs[i] = Inputs[i];
            }


            for (int i = 1; i < Layers.Length; i++) //forward prop
            {
                Layer currLayer = Layers[i];
                Layer prevLayer = Layers[i - 1];
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
            double[] Outputs = Layers[^1].Outputs;
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
        private static double[] ForwardProp(Layer[] Layers, double[] Inputs)
        {
            /*
             * 1) set first layer outputs as inputs to the forward prop - must be 1d array
             * 2) for each pair of curr and prev layer, calculate the necessary outputs of the next layers
             * 3) return resulting array of outputs
            */

            for (int i = 0; i < Inputs.Length; i++)
            {
                Layers[0].Outputs[i] = Inputs[i];
            }

            for (int i = 1; i < Layers.Length; i++) //forward prop
            {
                Layer currLayer = Layers[i];
                Layer prevLayer = Layers[i - 1];
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
            return Layers[^1].Outputs;
        }

        //  ---- BACKPROPAGATION ---- //
        // IMPORTANT!
        // Cost function C is referenced all throughout this backprop section, any cost functions can be used, but respective changes to the formulas must be made due to calculus chain rule differentiation
        // In this case it is: C = (a(n) - y)^2, or the Mean Squared Error (MSE) function, where a(n) = actual outputs, and y = expected outputs.
        private static void CalculateGradients(Layer[] Layers, double[] Target)
        {
            /*
             * 1) calculate last layer dCdZ and save it to the array
             * 2) for each pair of curr and next layers, calculate the previous dC/dZ (backpropagation)
             * FORMULAS:
             * 1) dC/dz(n) = 2(f(z(n)) - y) f'(z), where a(n) = f(z(n)) = final outputs
             * 2) dC/dA(n-1) = dC/dz(n) * w(n)
             * 3) dC/dZ(n-1) = dC/dA(n-1) * f'(z(n-1))
            */

            for (int i = 0; i < Layers[^1].NeuronCount; i++) //setting the dC/dZ(n) values of the last layer neurons using FORMULA 1
            {
                Layers[^1].dCdZ[i] = 2 * (Layers[^1].Outputs[i] - Target[i]) * Sigmoid_Deriv(Layers[^1].Outputs[i]); 
            }

            for (int LIndex = Layers.Length - 2; LIndex > 0; LIndex--)
            {
                Layer currLayer = Layers[LIndex]; //layer n - 1
                Layer nextLayer = Layers[LIndex + 1]; //layer n

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
        }
        private static void UpdateParameters(Layer[] Layers, double LearnRate)
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
            for (int LIndex = 1; LIndex < Layers.Length; LIndex++)
            {
                Layer currLayer = Layers[LIndex];
                Layer prevLayer = Layers[LIndex - 1];
                for (int currNeuron = 0; currNeuron < currLayer.NeuronCount; currNeuron++)
                {
                    for (int prevNeuron = 0; prevNeuron < prevLayer.NeuronCount; prevNeuron++)
                    {
                        currLayer.Weights[prevNeuron][currNeuron] -= LearnRate * currLayer.dCdZ[currNeuron] * prevLayer.Outputs[prevNeuron]; //updating weight between prev and curr neuron using dC/dw(n) by FORMULA 4
                        currLayer.Bias[currNeuron] -= LearnRate * currLayer.dCdZ[currNeuron]; //same as above but using FORMULA 5, so not multiplying by a(n - 1) as dz(n)/dB(n) = 1
                    }
                }
            }
        }
        // --- END BACKPROPAGATION --- //

        public static void TrainNetwork(NeuralNetwork Network, double[][] TrainData, double[][] TrainLabels, double[][] TestData, double[][] TestLabels, double LearnRate)
        {
            Stopwatch stopwatch = new Stopwatch();
            float prev_accuracy = GetPercentageAccuracy(Network, TestData, TestLabels);
            Console.WriteLine($"Epoch 0: initialisation. Initial performance: {prev_accuracy}.");
            for (int epoch = 1; epoch < 100; epoch++)
            {
                stopwatch.Start();
                for (int i = 0; i < TrainData.Length; i++)
                {
                    ForwardProp(Network.Layers, TrainData[i]);
                    CalculateGradients(Network.Layers, TrainLabels[i]);
                    UpdateParameters(Network.Layers, LearnRate);
                }
                stopwatch.Stop();
                TimeSpan temp = stopwatch.Elapsed;
                stopwatch.Restart();
                float accuracy = GetPercentageAccuracy(Network, TestData, TestLabels);
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
        public static float GetPercentageAccuracy(NeuralNetwork network, double[][] TestData, double[][] TestLabels)
        {
            int count = 0;
            double max = -1;
            int index = -1;
            for (int i = 0; i < TestData.Length; i++)
            {
                int Label = ForwardPropClassify(network.Layers, TestData[i]);

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


        // ------------- NEW OPTIMISED NETWORK CODE ---------------------------
        public static void TrainNetwork(OptimisedNetwork Network, double[][] TrainData, double[][] TrainLabels, double[][] TestData, double[][] TestLabels, double LearnRate)
        {
            Stopwatch stopwatch = new Stopwatch();
            float prev_accuracy = GetPercentageAccuracy(Network, TestData, TestLabels);
            Console.WriteLine($"Epoch 0: initialisation. Initial performance: {prev_accuracy}.");
            for (int epoch = 1; epoch < 100; epoch++)
            {
                stopwatch.Start();
                for (int i = 0; i < TrainData.Length; i++)
                {
                    OptForwardProp(Network, TrainData[i]);
                    OptCalculateGradients(Network, TrainLabels[i]);
                    OptUpdateParameters(Network, LearnRate);
                }
                stopwatch.Stop();
                TimeSpan temp = stopwatch.Elapsed;
                stopwatch.Restart();
                float accuracy = GetPercentageAccuracy(Network, TestData, TestLabels);
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

        public static double[] OptForwardProp(OptimisedNetwork Network, double[] Inputs)
        {
            /*
             * 1) set first layer outputs as inputs to the forward prop - must be 1d array
             * 2) for each pair of curr and prev layer, calculate the necessary outputs of the next layers
             * 3) return resulting array of outputs
            */

            Network.Outputs[0] = Inputs; //step 1

            for (int LayerIndex = 1; LayerIndex < Network.LayerCount; LayerIndex++)
            {
                for (int currNeuron = 0; currNeuron < Network.NeuronCounts[LayerIndex]; currNeuron++)
                {
                    double output = Network.Biases[LayerIndex - 1][currNeuron];
                    for (int prevNeuron = 0; prevNeuron < Network.NeuronCounts[LayerIndex - 1]; prevNeuron++)
                    {
                        output += Network.Outputs[LayerIndex - 1][prevNeuron] * Network.Weights[LayerIndex - 1][prevNeuron][currNeuron];
                    }
                    Network.Outputs[LayerIndex][currNeuron] = Sigmoid(output);
                }
            }
            return Network.Outputs[^1];
        }

        private static int OptForwardPropClassify(OptimisedNetwork Network, double[] Inputs)
        {
            /*
             * 1) set first layer outputs as inputs to the forward prop - must be 1d array
             * 2) for each pair of curr and prev layer, calculate the necessary outputs of the next layers
             * 3) return the index of the maximum value in the final output array (index of classification i.e. label)
            */

            Network.Outputs[0] = Inputs;

            for (int LayerIndex = 1; LayerIndex < Network.LayerCount; LayerIndex++)
            {
                for (int currNeuron = 0; currNeuron < Network.NeuronCounts[LayerIndex]; currNeuron++)
                {
                    double output = Network.Biases[LayerIndex - 1][currNeuron];
                    for (int prevNeuron = 0; prevNeuron < Network.NeuronCounts[LayerIndex - 1]; prevNeuron++)
                    {
                        output += Network.Outputs[LayerIndex - 1][prevNeuron] * Network.Weights[LayerIndex - 1][prevNeuron][currNeuron];
                    }
                    Network.Outputs[LayerIndex][currNeuron] = Sigmoid(output);
                }
            }

            //series of comparisons to get the index of the max value from the final layer outputs
            double max = -1;
            int index = -1;
            double[] Outputs = Network.Outputs[^1];
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

        private static void OptCalculateGradients(OptimisedNetwork Network, double[] Target)
        {
            /*
             * 1) calculate last layer dCdZ and save it to the array
             * 2) for each pair of curr and next layers, calculate the previous dC/dZ (backpropagation)
             * FORMULAS:
             * 1) dC/dz(n) = 2(f(z(n)) - y) f'(z), where a(n) = f(z(n)) = final outputs
             * 2) dC/dA(n-1) = dC/dz(n) * w(n)
             * 3) dC/dZ(n-1) = dC/dA(n-1) * f'(z(n-1))
            */

            for (int i = 0; i < Network.NeuronCounts[^1]; i++) //setting the dC/dZ(n) values of the last layer neurons using FORMULA 1
            {
                Network.dCdz[^1][i] = 2 * (Network.Outputs[^1][i] - Target[i]) * Sigmoid_Deriv(Network.Outputs[^1][i]);
            }


            for (int LayerIndex = Network.LayerCount - 2; LayerIndex > 0; LayerIndex--)
            {
                int NeuronCount = Network.NeuronCounts[LayerIndex];
                int NextNeuronCount = Network.NeuronCounts[LayerIndex + 1];
                for (int currNeuron = 0; currNeuron < NeuronCount; currNeuron++)
                {
                    double sum_dCdA = 0;
                    for (int nextNeuron = 0; nextNeuron < NextNeuronCount; nextNeuron++)
                    {
                        sum_dCdA += Network.Weights[LayerIndex - 1][currNeuron][nextNeuron] * Network.dCdz[LayerIndex + 1][nextNeuron]; //working out dC/da(n-1) using FORMULA 2
                    }
                    Network.dCdz[LayerIndex][currNeuron] = sum_dCdA * Sigmoid_Deriv(Network.Outputs[LayerIndex][currNeuron]); //working out dC/dZ(n-1) using FORMULA 3
                }
            }
        }
        private static void OptUpdateParameters(OptimisedNetwork Network, double LearnRate)
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

            for (int LayerIndex = 1; LayerIndex < Network.LayerCount; LayerIndex++)
            {
                int CurrNeuronCount = Network.NeuronCounts[LayerIndex];
                int PrevNeuronCount = Network.NeuronCounts[LayerIndex - 1];
                for (int currNeuron = 0; currNeuron < CurrNeuronCount; currNeuron++)
                {
                    for (int prevNeuron = 0; prevNeuron < PrevNeuronCount; prevNeuron++)
                    {
                        Network.Weights[LayerIndex - 1][prevNeuron][currNeuron] -= LearnRate * Network.dCdz[LayerIndex][currNeuron] * Network.Outputs[LayerIndex - 1][prevNeuron];  //updating weight between prev and curr neuron using dC/dw(n) by FORMULA 4
                        Network.Biases[LayerIndex - 1][currNeuron] -= LearnRate * Network.dCdz[LayerIndex][currNeuron]; //same as above but using FORMULA 5, so not multiplying by a(n - 1) as dz(n)/dB(n) = 1
                    }
                }
            }
        }

        public static float GetPercentageAccuracy(OptimisedNetwork network, double[][] TestData, double[][] TestLabels)
        {
            int count = 0;
            double max = -1;
            int index = -1;
            for (int i = 0; i < TestData.Length; i++)
            {
                int Label = OptForwardPropClassify(network, TestData[i]);

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

        public static double Sigmoid(double x)
        {
            if (x > 40)
            {
                return 1;
            }
            else if (x < -40)
            {
                return 0;
            }
            return 1 / (1 + Math.Exp(-x));
        }
        public static double Sigmoid_Deriv(double x)
        {
            if (x > 40 || x < -40)
            {
                return 0;
            }
            double y = Sigmoid(x); //y = sigmoid(x)
            return y * (1 - y); //d(sigmoidx)/dx = y(1-y)
        }
    }
}
