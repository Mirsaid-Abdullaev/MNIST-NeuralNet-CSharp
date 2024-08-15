using System.Diagnostics;
using System;
using static NeuralNetworks.MathUtil;

namespace NeuralNetworks
{
    internal class SGDMomentum : ISGDMAlgorithm
    {
        protected override int ForwardPropClassify(double[] Inputs)
        {
            /*
             * 1) set first layer outputs as inputs to the forward prop - must be 1d array
             * 2) for each pair of curr and prev layer, calculate the necessary outputs of the next layers
             * 3) return the index of the maximum value in the final output array (index of classification i.e. label)
            */
            var CurrNet = (MomentumNetwork) Network;

            for (int i = 0; i < Inputs.Length; i++)
            {
                CurrNet.Layers[0].Outputs[i] = Inputs[i];
            }


            for (int i = 1; i < CurrNet.Layers.Length; i++) //forward prop
            {
                MomentumLayer currLayer = CurrNet.Layers[i];
                MomentumLayer prevLayer = CurrNet.Layers[i - 1];
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
            var CurrNet = (MomentumNetwork)Network;

            for (int i = 0; i < Inputs.Length; i++)
            {
                CurrNet.Layers[0].Outputs[i] = Inputs[i];
            }
            int NumLayers = CurrNet.Layers.Length;
            for (int i = 1; i < NumLayers; i++) //forward prop
            {
                MomentumLayer currLayer = CurrNet.Layers[i];
                MomentumLayer prevLayer = CurrNet.Layers[i - 1];
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
            var CurrNet = (MomentumNetwork)Network;

            for (int i = 0; i < CurrNet.Layers[^1].NeuronCount; i++) //setting the dC/dZ(n) values of the last layer neurons using FORMULA 1
            {
                CurrNet.Layers[^1].dCdZ[i] = 2 * (CurrNet.Layers[^1].Outputs[i] - Target[i]) * Sigmoid_InvDeriv(CurrNet.Layers[^1].Outputs[i]);
            }

            for (int LIndex = CurrNet.Layers.Length - 2; LIndex > 0; LIndex--)
            {
                MomentumLayer currLayer = CurrNet.Layers[LIndex]; //layer n - 1
                MomentumLayer nextLayer = CurrNet.Layers[LIndex + 1]; //layer n

                for (int currNeuron = 0; currNeuron < currLayer.NeuronCount; currNeuron++)
                {
                    double Sum_dCdA = 0;
                    for (int nextNeuron = 0; nextNeuron < nextLayer.NeuronCount; nextNeuron++)
                    {
                        Sum_dCdA += nextLayer.Weights[currNeuron][nextNeuron] * nextLayer.dCdZ[nextNeuron]; //working out dC/da(n-1) using FORMULA 2
                    }
                    currLayer.dCdZ[currNeuron] = Sum_dCdA * Sigmoid_InvDeriv(currLayer.Outputs[currNeuron]); //working out dC/dZ(n-1) using FORMULA 3
                }
            }
            Network = CurrNet;
        }
        protected override void UpdateParameters(double LEARN_RATE)
        {
            /*
             * 1) for each pair of curr and prev layers, use curr dC/dZ(n) and prev a(n-1) to find dC/dW(n)
             * 2) calculate the new mw(n) and mb(n) values for each neuron/neuron pair using the derivatives
             * 3) use mw(n) * learnrate and subtract from current w(n)
             * 4) use dC/db(n) = dC/dZ(n), to subtract mb(n) * learnrate from b(n)
             * 
             * FORMULAS:
             * 4) dC/dw(n) = dC/dz(n) * a(n-1), n >= 1
             * 5) dC/dB(n) = dC/dz(n)
             * 6) new_w(x) = old_w(x) - learnrate * mw(x)
             * 7) new_b(x) = old_b(x) - learnrate * mb(x)
             * 8) mw(x) = momentum_rate * mw(x) + (1 - momentum_rate) * dC/dW(x)  
             * 9) mb(x) = momentum_rate * mb(x) + (1 - momentum_rate) * dC/dB(x)  
            */
            var CurrNet = (MomentumNetwork)Network;

            for (int LIndex = 1; LIndex < CurrNet.Layers.Length; LIndex++)
            {
                MomentumLayer currLayer = CurrNet.Layers[LIndex];
                MomentumLayer prevLayer = CurrNet.Layers[LIndex - 1];
                for (int currNeuron = 0; currNeuron < currLayer.NeuronCount; currNeuron++)
                {
                    for (int prevNeuron = 0; prevNeuron < prevLayer.NeuronCount; prevNeuron++)
                    {
                        double dCdW = currLayer.dCdZ[currNeuron] * prevLayer.Outputs[prevNeuron]; //dCdW calculation using FORMULA 4
                        double dCdB = currLayer.dCdZ[currNeuron]; //dCdB using FORMULA 5;
                        double WeightDelta = MOMENTUM_RATE * currLayer.MomentumWeights[prevNeuron][currNeuron] + (1 - MOMENTUM_RATE) * dCdW; //mw(x) calculation using FORMULA 8
                        double BiasDelta = MOMENTUM_RATE * currLayer.MomentumBias[currNeuron] + (1 - MOMENTUM_RATE) * dCdB; //mb(x) calculation using FORMULA 9

                        currLayer.MomentumWeights[prevNeuron][currNeuron] = WeightDelta;
                        currLayer.MomentumBias[currNeuron] = BiasDelta;

                        currLayer.Weights[prevNeuron][currNeuron] -= LEARN_RATE * WeightDelta; //updating weight between prev and curr neuron using dC/dw(n) by FORMULA 8
                        currLayer.Bias[currNeuron] -= LEARN_RATE * BiasDelta; //same as above but using FORMULA 9, so not multiplying by a(n - 1) as dz(n)/dB(n) = 1
                    }
                }
            }
            Network = CurrNet;
        }
        // --- END BACKPROPAGATION --- //

        public override void TrainNetwork(INetwork MomentumNet, double[][] TrainData, double[][] TrainLabels, double[][] TestData, double[][] TestLabels, double LEARN_RATE, int Epochs)
        {
            try
            {
                var temp = (MomentumNetwork)MomentumNet;
            }
            catch
            {
                throw new Exception("Error: current network instance is not of type MomentumNetwork.");
            }
            this.Network = (MomentumNetwork)MomentumNet;

            Stopwatch stopwatch = new Stopwatch();
            float best_accuracy = GetPercentageAccuracy(TestData, TestLabels);
            Console.WriteLine($"Epoch 0: initialisation. Initial performance: {best_accuracy}.");
            for (int epoch = 1; epoch < Epochs; epoch++)
            {
                stopwatch.Start();
                for (int i = 0; i < TrainData.Length; i++)
                {
                    ForwardProp(TrainData[i]);
                    CalculateGradients(TrainLabels[i]);
                    UpdateParameters(LEARN_RATE);
                }
                stopwatch.Stop();
                TimeSpan temp = stopwatch.Elapsed;
                stopwatch.Restart();
                float accuracy = GetPercentageAccuracy(TestData, TestLabels);
                if (accuracy > best_accuracy) //need to save network
                {
                    MomentumNet.SaveNetwork("TEST1");
                    best_accuracy = accuracy;
                }
                stopwatch.Stop();
                Console.WriteLine($"Epoch {epoch} completed. Current performance %: {accuracy}. Epoch training time: {temp}. Test run time: {stopwatch.Elapsed}");
                stopwatch.Reset();
            }
        }
        public override void BenchmarkConvergence(INetwork MomentumNet, double[][] TrainData, double[][] TrainLabels, double[][] TestData, double[][] TestLabels)
        {
            try
            {
                var temp = (MomentumNetwork)MomentumNet;
            }
            catch
            {
                throw new Exception("Error: current network instance is not of type MomentumNetwork.");
            }
            this.Network = (MomentumNetwork)MomentumNet;
            Stopwatch stopwatch = new Stopwatch();
            float best_accuracy = GetPercentageAccuracy(TestData, TestLabels);
            TimeSpan total_training = new TimeSpan(0);
            TimeSpan total_validating = new TimeSpan(0);
            float initial_accuracy = best_accuracy;
            float accuracy = best_accuracy;
            Console.WriteLine($"Epoch 0: initialisation. Initial performance: {initial_accuracy}.");
            for (int epoch = 1; epoch < 100; epoch++)
            {
                stopwatch.Start();
                for (int i = 0; i < TrainData.Length; i++)
                {
                    ForwardProp(TrainData[i]);
                    CalculateGradients(TrainLabels[i]);
                    UpdateParameters(0.005);
                }
                stopwatch.Stop();
                TimeSpan temp = stopwatch.Elapsed;
                total_training += temp;
                stopwatch.Restart();
                accuracy = GetPercentageAccuracy(TestData, TestLabels);
                if (accuracy > best_accuracy) //need to save network
                {
                    MomentumNet.SaveNetwork("TEST1");
                    best_accuracy = accuracy;
                }
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
            Console.WriteLine($"Average time to run a validation epoch: {total_validating / 100}");
        }
    }
}
