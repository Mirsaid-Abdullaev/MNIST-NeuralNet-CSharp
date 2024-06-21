using System;
using System.IO;
using System.Text;

namespace NeuralNetwork
{
    internal class NeuralNetwork
    {
        public Layer[] Layers;
        public NeuralNetwork(int[] LayersData) //LayersData holds number of neurons in each layer in the network e.g. {2, 3, 1}, {100, 150, 50, 10, 8} etc.
        {
            Layers = new Layer[LayersData.Length];
            Layers[0] = new Layer(LayersData[0], 0);
            for (int i = 1; i < Layers.Length; i++)
            {
                Layers[i] = new Layer(LayersData[i], LayersData[i - 1]);
            }
        }

        public NeuralNetwork(Layer[] Layers) //for loading from storage
        {
            this.Layers = Layers;
        }

        public void SaveNetwork(string FileName)
        {
            string FilePath = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData) + @"\MNIST_Net\" + FileName + ".csv";
            if (!Directory.Exists(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData) + @"\MNIST_Net\"))
            {
                Directory.CreateDirectory(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData) + @"\MNIST_Net\");
            }
            using StreamWriter sw = new StreamWriter(FilePath, false);
            sw.Write(GetNetworkData());
            sw.Flush();
        }

        private string GetNetworkData()
        {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.AppendLine(Layers.Length.ToString());
            for (int i = 0;i < Layers.Length - 1; i++)
            {
                stringBuilder.AppendLine(Layers[i].GetLayerData());
            }
            stringBuilder.Append(Layers[^1].GetLayerData());
            return stringBuilder.ToString();
        }
    }
}
