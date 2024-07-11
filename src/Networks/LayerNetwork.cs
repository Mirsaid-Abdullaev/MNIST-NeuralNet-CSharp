using System;
using System.IO;
using System.Text;

namespace NeuralNetworks
{
    internal class LayerNetwork: INetwork
    {
        public Layer[] Layers;
        public LayerNetwork(int[] LayersData) //LayersData holds number of neurons in each layer in the network e.g. {2, 3, 1}, {100, 150, 50, 10, 8} etc.
        {
            Layers = new Layer[LayersData.Length];
            Layers[0] = new Layer(LayersData[0], 0);
            for (int i = 1; i < Layers.Length; i++)
            {
                Layers[i] = new Layer(LayersData[i], LayersData[i - 1]);
            }
        }

        public LayerNetwork(Layer[] Layers) //for loading from storage
        {
            this.Layers = Layers;
        }

        public override void SaveNetwork(string FileName)
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

        protected override string GetNetworkData()
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
