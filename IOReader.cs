﻿using System;
using System.IO;

namespace NeuralNetwork
{
    internal static class IOReader
    {
        static readonly string projectDirectory = "D:\\NeuralNetwork";

        private static double[] CreateOneHot(byte Label)
        {
            double[] Result = null;
            if (Label > 9)
            {
                throw new Exception("Error: label number is not within 0-9, cannot create a 10-length One Hot formatted array from this digit.");
            }
            switch (Label)
            {
                case 0:
                    Result = new double[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                    break;
                case 1:
                    Result = new double[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
                    break;
                case 2:
                    Result = new double[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
                    break;
                case 3:
                    Result = new double[] { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 };
                    break;
                case 4:
                    Result = new double[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 };
                    break;
                case 5:
                    Result = new double[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };
                    break;
                case 6:
                    Result = new double[] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 };
                    break;
                case 7:
                    Result = new double[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 };
                    break;
                case 8:
                    Result = new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 };
                    break;
                case 9:
                    Result = new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
                    break;
            }

            return Result;
        }
        public static double[][] GetTestDataOutputs(int max = 0) //array of 10,000 image labels stored sequentially in one-hot encoding
        {
            double[][] Outputs;


            using (FileStream ifsLabels = new FileStream(projectDirectory + @"\MNIST\t10k-labels.idx1-ubyte", FileMode.Open))
            {
                using BinaryReader brLabels = new BinaryReader(ifsLabels);
                brLabels.ReadBigInt32(); //magic int, not required
                int numLabels = brLabels.ReadBigInt32();
                if (max != 0)
                {
                    numLabels = max;
                }

                Outputs = new double[numLabels][];

                for (int i = 0; i < numLabels; i++)
                {
                    Outputs[i] = CreateOneHot(brLabels.ReadByte());
                }
            }
            return Outputs;
        }
        public static double[][] GetTrainingDataOutputs(int max = 0)
        {
            double[][] Outputs;

            using (FileStream ifsLabels = new FileStream(projectDirectory + @"\MNIST\train-labels.idx1-ubyte", FileMode.Open))
            {
                using BinaryReader brLabels = new BinaryReader(ifsLabels);
                brLabels.ReadBigInt32(); //magic int, no purpose
                int numLabels = brLabels.ReadBigInt32();
                if (max != 0)
                {
                    numLabels = max;
                }

                Outputs = new double[numLabels][];

                for (int i = 0; i < numLabels; i++)
                {
                    Outputs[i] = CreateOneHot(brLabels.ReadByte());
                }
            }

            return Outputs;
        }
        public static double[][] GetTrainingDataInputs(int max = 0)
        {
            double[][] Outputs;

            using (FileStream ifsImages = new FileStream(projectDirectory + @"\MNIST\train-images.idx3-ubyte", FileMode.Open))
            {
                using BinaryReader brImages = new BinaryReader(ifsImages);
                brImages.ReadBigInt32(); //a magic number int, dont need this
                int numImages = brImages.ReadBigInt32();
                brImages.ReadBigInt32(); //num of rows, dont need this
                brImages.ReadBigInt32(); //nu of cols, dont need this
                if (max != 0)
                {
                    numImages = max;
                }
                Outputs = new double[numImages][];

                for (int i = 0; i < numImages; i++)
                {
                    Outputs[i] = new double[784];
                    for (int j = 0; j < 784; j++)
                    {
                        Outputs[i][j] = (double)brImages.ReadByte() / 255;
                    }
                }
            }
            return Outputs; //outputs stores images in sequential order like [image1, image2, image3...] where image = [0, 1, 2, ..., 783] each pixel value
        }
        public static double[][] GetTestDataInputs(int max = 0)
        {
            double[][] Outputs; //each row is an image

            using (FileStream ifsImages = new FileStream(projectDirectory + @"\MNIST\t10k-images.idx3-ubyte", FileMode.Open))
            {
                using BinaryReader brImages = new BinaryReader(ifsImages);
                brImages.ReadBigInt32(); //a magic number int, dont need this
                int numImages = brImages.ReadBigInt32();
                brImages.ReadBigInt32(); //num of rows, dont need this
                brImages.ReadBigInt32(); //nu of cols, dont need this
                if (max != 0)
                {
                    numImages = max;
                }
                Outputs = new double[numImages][];


                for (int i = 0; i < numImages; i++)
                {
                    Outputs[i] = new double[784];
                    for (int j = 0; j < 784; j++)
                    {
                        Outputs[i][j] = (double)brImages.ReadByte() / 255;
                    }
                }
            }

            return Outputs;
        }

        public static NeuralNetwork LoadNetwork(string FilePath)
        {
            if (!File.Exists(FilePath))
            {
                throw new Exception("Error: no file found at the specified location.");
            }
            try
            {
                string[] NetData = ParseFileCsv(FilePath);
                int NoOfLayers = Convert.ToInt32(NetData[0]);
                Layer[] Layers = new Layer[NoOfLayers];
                string[] LayerData = NetData[1].Split(',');
                Layers[0] = new Layer(Convert.ToInt32(LayerData[0]), 0);

                //input layer has no prev neurons and no weights matrix or bias
                for (int i = 2; i <= NoOfLayers; i++)
                {
                    LayerData = NetData[i].Split(',');
                    int currNeurons = Convert.ToInt32(LayerData[0]);
                    int prevNeurons = Convert.ToInt32(LayerData[1]);

                    double[][] Weights = new double[prevNeurons][];
                    double[] Bias = new double[currNeurons];
                    int currRow = -1;
                    for (int j = 2; j < 2 + prevNeurons * currNeurons; j++)
                    {
                        int col = (j - 2) % currNeurons;
                        if (col == 0) //creating a new row
                        {
                            currRow++;
                            Weights[currRow] = new double[currNeurons];
                            Weights[currRow][col] = Convert.ToDouble(LayerData[j]);
                        }
                        else //on the same row
                        {
                            Weights[currRow][col] = Convert.ToDouble(LayerData[j]);
                        }
                    }
                    for (int j = 2 + prevNeurons * currNeurons; j < LayerData.Length; j++)
                    {
                        Bias[j - 2 - prevNeurons * currNeurons] = Convert.ToDouble(LayerData[j]);
                    }
                    Layer layer = new Layer(currNeurons, prevNeurons, Weights, Bias);
                    Layers[i - 1] = layer;
                }
                return new NeuralNetwork(Layers);
            }
            catch
            {
                throw new Exception("Error: file was not a correctly formatted NeuralNetwork file.");
            }
        }
        private static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(int));
            if (BitConverter.IsLittleEndian)
                Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }
        private static string[] ParseFileCsv(string FilePath) //returns all the rows of a csv file
        {
            string[] FullData;
            using (StreamReader reader = new StreamReader(FilePath))
            {
                try
                {
                    FullData = reader.ReadToEnd().Split('\n', StringSplitOptions.RemoveEmptyEntries);
                }
                catch
                {
                    return null;
                }
            }
            return FullData; //returns all not-null data rows
        }
    }
}
