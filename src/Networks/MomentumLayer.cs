﻿using System.Text;
using System;

namespace NeuralNetworks
{
    internal class MomentumLayer
    {
        public int NeuronCount; //number of neurons in the current layer
        public int PrevNeuronCount; //number of neurons in prev layer

        public readonly double[] Bias; //Each index relates to each neuron in this layer (b(n))
        public readonly double[] Outputs; //As above, but each index is the same neuron's output value (a(n) = f(z(n)))
        public readonly double[] dCdZ; //As above, but each index is the same neuron's dC/dZ value (dC/dZ(n) = dC/da(n) * da(n)/dz(n) = 2 * (a(n) - y) * f'(z(n)))
        public readonly double[][] Weights;

        // ADDITION TO THE LAYER CLASS - THESE ARRAYS HOLD THE EXPONENTIAL MOVING AVERAGES OF THE PREVIOUS DC/DW VALUES //
        public readonly double[][] MomentumWeights; //mw(n) = momentum_rate * mw(n - 1) + (1 - momentum_rate) * dC/dw(n), mw(0) = 0
        public readonly double[] MomentumBias; //mb(n) = momentum_rate * mb(n - 1) + (1 - momentum_rate) * dC/db(n), mb(0) = 0

        public MomentumLayer(int Neurons, int PrevNeurons) //setting all weights and biases for all neurons in one go
        {
            NeuronCount = Neurons;
            PrevNeuronCount = PrevNeurons;

            if (PrevNeuronCount > 0) //checking not input layer
            {
                double temp = Math.Sqrt(PrevNeuronCount);

                Weights = new double[PrevNeuronCount][];
                Bias = new double[NeuronCount];
                dCdZ = new double[NeuronCount];

                MomentumWeights = new double[PrevNeuronCount][];
                MomentumBias = new double[NeuronCount];

                for (int i = 0; i < PrevNeuronCount; i++)
                {
                    Weights[i] = new double[NeuronCount];
                    MomentumWeights[i] = new double[NeuronCount];
                    for (int j = 0; j < NeuronCount; j++)
                    {
                        Weights[i][j] = GetRandom() / temp;
                    }
                }
                for (int i = 0; i < NeuronCount; i++)
                {
                    Bias[i] = GetRandom() / temp;
                }
            }
            Outputs = new double[NeuronCount]; //all layers have output layer, as input layer has a(0) which are just the network inputs
        }
        public MomentumLayer(int Neurons, int PrevNeurons, double[][] Weights, double[] Bias) //for loading a layer from storage
        {
            NeuronCount = Neurons;
            PrevNeuronCount = PrevNeurons;

            if (PrevNeuronCount > 0) //not input layer
            {
                this.Weights = Weights;
                this.Bias = Bias;
                dCdZ = new double[NeuronCount];
                MomentumWeights = new double[PrevNeuronCount][];
                MomentumBias = new double[NeuronCount];
                for (int i = 0; i < PrevNeuronCount; i++)
                {
                    MomentumWeights[i] = new double[NeuronCount];
                }
            }
            Outputs = new double[NeuronCount];
        }

        static double GetRandom() //y = sqrt(-2 * ln(1 - x1)) * cos(2 * pi * x2), 0 < x1 <= 1, 0 < x2 <= 1
        {
            Random random = new Random();
            return Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble());
        }

        public string GetLayerData()
        {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.Append(NeuronCount.ToString() + "," + PrevNeuronCount.ToString() + ",");
            if (PrevNeuronCount != 0)
            {
                for (int i = 0; i < PrevNeuronCount; i++)
                {
                    stringBuilder.Append(string.Join(",", Weights[i]) + ",");
                }
                stringBuilder.Append(string.Join(",", Bias));
            }
            return stringBuilder.ToString();
        }
    }
}
