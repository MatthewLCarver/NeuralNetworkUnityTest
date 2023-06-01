using System.Collections.Generic;
using UnityEngine;

namespace NeuralNet
{
    [CreateAssetMenu(fileName = "NeuralNetworkData", menuName = "Neural Network/Neural Network Data", order = 0)]
    public class NeuralNetData : ScriptableObject
    {
        [SerializeField,
        Tooltip("The total number of inputs of the neural network.")]
        private int inputCount;

        [SerializeField,
        Tooltip("The total number of outputs of the neural network.")] 
        private int outputCount;
        
        [SerializeField,
        Tooltip("The total number of hidden layers of the neural network.")]
        private int hiddenLayerCount;
        
        [SerializeField,
        Tooltip("The total number of neurons in each hidden layer of the neural network.")]
        private List<int> hiddenNeuronList;
        
        [SerializeField,
        Tooltip("The activation type of the neural network.")]
        private ActivationType activationType;
        
        [SerializeField,
        Tooltip("The learning rate of the neural network.")]
        private float learningRate = 0.1f;
        
        /// <summary>
        /// Sets the data of the neural network when data is individually given.
        /// </summary>
        /// <param name="_inputCount"></param>
        /// <param name="_hiddenLayerCount"></param>
        /// <param name="_hiddenNeuronCountList"></param>
        /// <param name="_outputCount"></param>
        /// <param name="_activationType"></param>
        /// <param name="_learningRate"></param>
        public void SetNeuralNetworkData(int _inputCount, int _hiddenLayerCount, List<int> _hiddenNeuronCountList, 
            int _outputCount, ActivationType _activationType, float _learningRate = 0.1f)
        {
            inputCount = _inputCount;
            hiddenLayerCount = _hiddenLayerCount;
            hiddenNeuronList = _hiddenNeuronCountList;
            outputCount = _outputCount;
            activationType = _activationType;
            learningRate = _learningRate;
        }
        
        /// <summary>
        /// Returns the data of the neural network as a NNData struct.
        /// </summary>
        /// <returns></returns>
        public NNData GetNeuralNetworkData()
        {
            NNData nnData = new NNData();
            nnData.inputCount = inputCount;
            nnData.hiddenLayerCount = hiddenLayerCount;
            nnData.hiddenNeuronList = hiddenNeuronList;
            nnData.outputCount = outputCount;
            nnData.activationType = activationType;
            nnData.learningRate = learningRate;
            return nnData;
        }
        
        /// <summary>
        /// Returns the learning rate of the neural network.
        /// </summary>
        /// <returns></returns>
        public float GetLearningRate()
        {
            return learningRate;
        }
        
        /// <summary>
        /// Returns the activation type of the neural network.
        /// </summary>
        /// <returns></returns>
        public ActivationType GetActivationType()
        {
            return activationType;
        }
        
        /// <summary>
        /// Sets the input count of the neural network.
        /// </summary>
        /// <param name="_inputCount"></param>
        public void SetInputCount(int _inputCount) => inputCount = _inputCount;
        
        /// <summary>
        /// Returns the input count of the neural network as an integer.
        /// </summary>
        /// <returns></returns>
        public int GetInputCount() => inputCount;
        
        /// <summary>
        /// Returns the hidden layer count of the neural network as an integer.
        /// </summary>
        /// <returns></returns>
        public int GetHiddenLayerCount() => hiddenLayerCount;
        
        /// <summary>
        /// Returns the hidden neuron list of the neural network as a list of integers.
        /// </summary>
        /// <returns></returns>
        public List<int> GetHiddenNeuronList() => hiddenNeuronList;
        
        /// <summary>
        /// Returns the output count of the neural network as an integer.
        /// </summary>
        /// <returns></returns>
        public int GetOutputCount() => outputCount;
        
        /// <summary>
        /// Sets the hidden layer count of the neural network.
        /// </summary>
        /// <param name="_hiddenLayerListCount"></param>
        public void SetHiddenLayerListCount(List<int> _hiddenLayerListCount)
        {
            hiddenNeuronList = _hiddenLayerListCount;
            hiddenLayerCount = hiddenNeuronList.Count;
        }

        /// <summary>
        /// Returns the hidden layer neuron count of the neural network as an array of integers.
        /// </summary>
        /// <returns></returns>
        public int[] GetHiddenLayerNeuronCount()
        {
            int[] hiddenLayerNeuronCount = new int[hiddenLayerCount];
            for (int i = 0; i < hiddenLayerCount; i++)
            {
                hiddenLayerNeuronCount[i] = hiddenNeuronList[i];
            }

            return hiddenLayerNeuronCount;
        }

        /// <summary>
        /// Sets the output count of the neural network.
        /// </summary>
        /// <param name="_outputCount"></param>
        public void SetOutputCount(int _outputCount) => outputCount = _outputCount;
    }

    /// <summary>
    /// A public struct that holds the data of the neural network.
    /// </summary>
    public struct NNData
    {
        public int inputCount;
        public int outputCount;
        public int hiddenLayerCount;
        public List<int> hiddenNeuronList;
        public ActivationType activationType;
        public float learningRate;
    }
}