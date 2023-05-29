using System.Collections.Generic;
using UnityEngine;

namespace NeuralNet
{
    [CreateAssetMenu(fileName = "NeuralNetworkData", menuName = "Neural Network/Neural Network Data", order = 0)]
    public class NeuralNetData : ScriptableObject
    {
        [SerializeField]
        private int inputCount;

        [SerializeField] 
        private int outputCount;
        
        [SerializeField]
        private int hiddenLayerCount;
        
        [SerializeField]
        private List<int> hiddenNeuronList;
        
        [SerializeField]
        private ActivationType activationType;
        
        [SerializeField]
        private float learningRate = 0.1f;
        
        
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
        
        public float GetLearningRate()
        {
            return learningRate;
        }
        
        public void SetInputCount(int _inputCount) => inputCount = _inputCount;

        public void SetHiddenLayerListCount(List<int> _hiddenLayerListCount)
        {
            hiddenNeuronList = _hiddenLayerListCount;
            hiddenLayerCount = hiddenNeuronList.Count;
        }

        public void SetOutputCount(int _outputCount) => outputCount = _outputCount;
    }

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