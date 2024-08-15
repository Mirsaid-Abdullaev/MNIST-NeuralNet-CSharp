using System.Diagnostics;
using System;
using static NeuralNetworks.MathUtil;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    internal class MiniBatchGD : IMBGDAlgorithm
    {
        private const int MINI_BATCH_SIZE = 32;
        protected override void ForwardProp(double[][] Inputs)
        {
            /*
             * 1) set first layer outputs as inputs to the forward prop - size of the inputs must be MINI_BATCH_SIZE x IMAGE_SIZE (in our case 32 x 784)
             * 2) for each pair of curr and prev layer, calculate the necessary outputs of the next layers FOR EVERY IMAGE in the mini batch
             * 3) return resulting 2D array of outputs 
            */
            var CurrNet = (MiniBatchNetwork)Network;

            Array.Copy(Inputs, CurrNet.Layers[0].Outputs, MINI_BATCH_SIZE); //setting every image to be a row in the Outputs array of the first (input) layer
            //copying Inputs[][] ==> Layers[0].Outputs[][]

            int NumLayers = CurrNet.Layers.Length;
            for (int i = 1; i < NumLayers; i++) //forward prop
            {
                MiniBatchLayer currLayer = CurrNet.Layers[i];
                MiniBatchLayer prevLayer = CurrNet.Layers[i - 1];
                Parallel.For(0, MINI_BATCH_SIZE, image =>
                {
                    for (int currNeuron = 0; currNeuron < currLayer.NeuronCount; currNeuron++)
                    {
                        double output = currLayer.Bias[currNeuron];
                        for (int prevNeuron = 0; prevNeuron < prevLayer.NeuronCount; prevNeuron++) //for each neuron in the previous letter
                        {
                            output += prevLayer.Outputs[image][prevNeuron] * currLayer.Weights[prevNeuron][currNeuron];
                        }
                        currLayer.Outputs[image][currNeuron] = Sigmoid(output);
                    }
                });
            }
            Network = CurrNet;
            return;
        }
        protected override int ForwardPropClassify(double[] Inputs) //this is still a per-image basis function
        {
            /*
             * 1) set first layer outputs as inputs to the forward prop
             * 2) for each pair of curr and prev layer, calculate the necessary outputs of the next layers
             * 3) return the indices of the maximum values in the final 2d output array for each image (index of classification i.e. label)
            */
            var CurrNet = (MiniBatchNetwork)Network;

            CurrNet.Layers[0].Outputs[0] = Inputs;

            int NumLayers = CurrNet.Layers.Length;
            for (int i = 1; i < NumLayers; i++)
            {
                MiniBatchLayer currLayer = CurrNet.Layers[i];
                MiniBatchLayer prevLayer = CurrNet.Layers[i - 1];
                
                for (int currNeuron = 0; currNeuron < currLayer.NeuronCount; currNeuron++)
                {
                    double output = currLayer.Bias[currNeuron];
                    for (int prevNeuron = 0; prevNeuron < prevLayer.NeuronCount; prevNeuron++) //for each neuron in the previous letter
                    {
                        output += prevLayer.Outputs[0][prevNeuron] * currLayer.Weights[prevNeuron][currNeuron];
                    }
                    currLayer.Outputs[0][currNeuron] = Sigmoid(output);
                }
            }
            
            //series of comparisons to get the index of the max value from the final layer outputs

            double[] Outputs = CurrNet.Layers[^1].Outputs[0];

            double max = -1;
            int index = -1;
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
        //  ---- BACKPROPAGATION ---- //
        // IMPORTANT!
        // Cost function C is referenced all throughout this backprop section, any cost functions can be used, but respective changes to the formulas must be made due to calculus chain rule differentiation
        // In this case it is: C = (a(n) - y)^2, or the Mean Squared Error (MSE) function, where a(n) = actual outputs, and y = expected outputs.
        protected override void CalculateGradients(double[][] Target)
        {
            /*
             * 1) calculate last layer dCdZ and save it to the array
             * 2) for each pair of curr and next layers, calculate the previous dC/dZ (backpropagation)
             * FORMULAS:
             * 1) dC/dz(n) = 1 / MINI_BATCH_SIZE * 2(f(z(n)) - y) f'(z), where a(n) = f(z(n)) = final outputs
             * 2) dC/dA(n-1) = dC/dz(n) * w(n)
             * 3) dC/dZ(n-1) = dC/dA(n-1) * f'(z(n-1))
            */
            var CurrNet = (MiniBatchNetwork)Network;
            Array.Copy(new double[CurrNet.Layers[^1].NeuronCount], CurrNet.Layers[^1].dCdZ, CurrNet.Layers[^1].NeuronCount); //resetting dCdZ

            Parallel.For(0, MINI_BATCH_SIZE, image =>
            {
                for (int i = 0; i < CurrNet.Layers[^1].NeuronCount; i++) //setting the dC/dZ(n) values of the last layer neurons using FORMULA 1
                {
                    CurrNet.Layers[^1].dCdZ[i] += 2 * (CurrNet.Layers[^1].Outputs[image][i] - Target[image][i]) * Sigmoid_InvDeriv(CurrNet.Layers[^1].Outputs[image][i]);
                }
            });
            for (int i = 0; i < CurrNet.Layers[^1].NeuronCount; i++)
            {
                CurrNet.Layers[^1].dCdZ[i] /= MINI_BATCH_SIZE; //performing an averaging for the dC/dZ values due to the usage of multiple images in one go
            }

            for (int LIndex = CurrNet.Layers.Length - 2; LIndex > 0; LIndex--)
            {
                MiniBatchLayer currLayer = CurrNet.Layers[LIndex]; //layer n - 1
                MiniBatchLayer nextLayer = CurrNet.Layers[LIndex + 1]; //layer n

                Array.Copy(new double[currLayer.NeuronCount], currLayer.dCdZ, CurrNet.Layers[^1].NeuronCount); //resetting dCdZ for the previous layer


                Parallel.For(0, MINI_BATCH_SIZE, image =>
                {
                    for (int currNeuron = 0; currNeuron < currLayer.NeuronCount; currNeuron++)
                    {
                        double Sum_dCdA = 0;
                        for (int nextNeuron = 0; nextNeuron < nextLayer.NeuronCount; nextNeuron++)
                        {
                            Sum_dCdA += nextLayer.Weights[currNeuron][nextNeuron] * nextLayer.dCdZ[nextNeuron]; //working out dC/da(n-1) using FORMULA 2
                        }
                        currLayer.dCdZ[currNeuron] += Sum_dCdA * Sigmoid_InvDeriv(currLayer.Outputs[image][currNeuron]); //working out dC/dZ(n-1) using FORMULA 3
                    }
                });
                for (int i = 0; i < CurrNet.Layers[^1].NeuronCount; i++)
                {
                    CurrNet.Layers[^1].dCdZ[i] /= MINI_BATCH_SIZE; //performing an averaging for the dC/dZ values due to the usage of multiple images in one go
                }
            }
            Network = CurrNet;
        }
        protected override void UpdateParameters(double LEARN_RATE)
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
            var CurrNet = (MiniBatchNetwork)Network;

            for (int LIndex = 1; LIndex < CurrNet.Layers.Length; LIndex++)
            {
                MiniBatchLayer currLayer = CurrNet.Layers[LIndex];
                MiniBatchLayer prevLayer = CurrNet.Layers[LIndex - 1];

                for (int currNeuron = 0; currNeuron < currLayer.NeuronCount; currNeuron++)
                {
                    for (int prevNeuron = 0; prevNeuron < prevLayer.NeuronCount; prevNeuron++)
                    {
                        double dCdW = 0;
                        double dCdB = 0;
                        Parallel.For(0, MINI_BATCH_SIZE, image => 
                        {
                            dCdW += currLayer.dCdZ[currNeuron] * prevLayer.Outputs[image][prevNeuron]; //dC/dW calculation using FORMULA 4
                            dCdB += currLayer.dCdZ[currNeuron]; //dCdB using FORMULA 5;
                        });
                        currLayer.Weights[prevNeuron][currNeuron] -= LEARN_RATE * dCdW; //updating weight between prev and curr neuron using dC/dw(n) by FORMULA 4 and by FORMULA 6
                        currLayer.Bias[currNeuron] -= LEARN_RATE * dCdB; //same as above but using FORMULA 5 and FORMULA 7, so not multiplying by a(n - 1) as dz(n)/dB(n) = 1
                    }
                }
            }
            Network = CurrNet;
        }
        // --- END BACKPROPAGATION --- //

        public override void TrainNetwork(INetwork LayerNet, double[][] TrainData, double[][] TrainLabels, double[][] TestData, double[][] TestLabels, double LearnRate, int Epochs)
        {
            try
            {
                var temp = (MiniBatchNetwork)LayerNet;
            }
            catch
            {
                throw new Exception("Error: current network instance is not of type MiniBatchNetwork.");
            }
            this.Network = (MiniBatchNetwork)LayerNet;

            Stopwatch stopwatch = new Stopwatch();
            float prev_accuracy = GetPercentageAccuracy(TestData, TestLabels);
            Console.WriteLine($"Epoch 0: initialisation. Initial performance: {prev_accuracy}.");
            for (int epoch = 1; epoch < Epochs; epoch++)
            {
                stopwatch.Start();
                int NumBatches = TrainData.Length % MINI_BATCH_SIZE == 0 ? TrainData.Length / MINI_BATCH_SIZE : TrainData.Length / MINI_BATCH_SIZE + 1;

                for (int i = 0; i < NumBatches; i++)
                {
                    ForwardProp(GetNextBatch(i * MINI_BATCH_SIZE, TrainData, 784));
                    CalculateGradients(GetNextBatch(i * MINI_BATCH_SIZE, TrainLabels, 10));
                    UpdateParameters(LearnRate);
                }
                stopwatch.Stop();
                TimeSpan temp = stopwatch.Elapsed;
                stopwatch.Restart();
                float accuracy = GetPercentageAccuracy(TestData, TestLabels);
                if (accuracy > prev_accuracy) //need to save network
                {
                    LayerNet.SaveNetwork("TEST1");
                }
                prev_accuracy = accuracy;
                stopwatch.Stop();
                Console.WriteLine($"Epoch {epoch} completed. Current performance %: {accuracy}. Epoch training time: {temp}. Test run time: {stopwatch.Elapsed}");
                stopwatch.Reset();
            }
        }

        public override void BenchmarkConvergence(INetwork MBGDNet, double[][] TrainData, double[][] TrainLabels, double[][] TestData, double[][] TestLabels)
        {
            try
            {
                var temp = (MiniBatchNetwork)MBGDNet;
            }
            catch
            {
                throw new Exception("Error: current network instance is not of type MiniBatchNetwork.");
            }
            this.Network = (MiniBatchNetwork)MBGDNet;

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
                    ForwardProp(GetNextBatch(i * MINI_BATCH_SIZE, TrainData, 784));
                    CalculateGradients(GetNextBatch(i * MINI_BATCH_SIZE, TrainLabels, 10));
                    UpdateParameters(0.01);
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
            Console.WriteLine($"Average time to run a validation epoch: {total_validating / 100}");
        }
        private double[][] GetNextBatch(int fromIndex, double[][] Data, int ItemLength) 
        {
            double[][] Batch = new double[MINI_BATCH_SIZE][];
            if (MINI_BATCH_SIZE > Data.Length - fromIndex)
            {
                Array.Copy(Data, fromIndex, Batch, 0, Data.Length - fromIndex);
                for (int i = 0; i < MINI_BATCH_SIZE + fromIndex - Data.Length; i++)
                {
                    Batch[i] = new double[ItemLength]; //filling the rest of the batch with 0-values if the source array had less than 32 images in the last batch
                    for (int j = 0; j < Batch[i].Length; j++)
                    {
                        Batch[i][j] = 0;
                    }
                }
            }
            else
            {
                Array.Copy(Data, fromIndex, Batch, 0, MINI_BATCH_SIZE);
            }
            return Batch;
        }
    }

    
}
