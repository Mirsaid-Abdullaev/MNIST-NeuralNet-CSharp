namespace NeuralNetworks
{
    internal abstract class INetwork
    {
        public abstract void SaveNetwork(string FileName);
        protected abstract string GetNetworkData();
    }
}
