using System;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;

using SaveLoad;

namespace NeuralNet
{
    [CreateAssetMenu(fileName = "TrainingData", menuName = "Neural Network/Training Data", order = 0)]
    public class TrainingData : ScriptableObject
    {
        [SerializeField,
        Tooltip("Contains the float data of the neural network.")]
        private TrainingDataStruct data;
        
        /// <summary>
        /// Contains the base data of the neural network. 
        /// </summary>
        private NeuralNetData nnData;
        
        /// <summary>
        /// The structured model data of the neural network that can be saved and loaded.
        /// </summary>
        [SerializeField]
        private TrainingModel trainingModel;
        
        /// <summary>
        /// The agent that is using this training data.
        /// </summary>
        private TrainableAgent agent;
        
        /// <summary>
        /// The name of the file that will be saved.
        /// </summary>
        private string fileName = "TrainedModel";

        /// <summary>
        /// Sets the data of the neural network when data is individually given.
        /// </summary>
        /// <param name="_inputLayer"></param>
        /// <param name="_hiddenLayers"></param>
        /// <param name="_outputLayer"></param>
        /// <param name="_weights"></param>
        /// <param name="_biases"></param>
        /// <param name="_activationType"></param>
        /// <param name="_learningRate"></param>
        /// <param name="_agent"></param>
        public void SetTrainingData(Matrix<float> _inputLayer, List<Matrix<float>> _hiddenLayers, 
                                    Matrix<float> _outputLayer, List<Matrix<float>> _weights, List<Matrix<float>> _biases,
                                    ActivationType _activationType, float _learningRate = 0.1f, TrainableAgent _agent = null)
        {
            agent = _agent;
            data.SetData(_inputLayer, _hiddenLayers, _outputLayer, _weights, _biases, _learningRate, _activationType);
        }
        
        /// <summary>
        /// Sets the data of the neural network when data is given as a TrainingDataStruct.
        /// </summary>
        /// <param name="_data"></param>
        /// <param name="_fitness"></param>
        /// <param name="_agent"></param>
        public void SetTrainingData(TrainingDataStruct _data, float _fitness, TrainableAgent _agent = null)
        {
            agent = _agent;
            data.SetData(_data.inputLayer, _data.hiddenLayers, _data.outputLayer, _data.weights, _data.biases,
                _fitness, _data.activationType);
        }
        
        /// <summary>
        /// Returns the data of the neural network when passed as a reference.
        /// </summary>
        /// <param name="_inputLayer"></param>
        /// <param name="_hiddenLayers"></param>
        /// <param name="_outputLayer"></param>
        /// <param name="_weights"></param>
        /// <param name="_biases"></param>
        /// <param name="_fitness"></param>
        /// <param name="_activationType"></param>
        public void GetTrainingData(out Matrix<float> _inputLayer, out List<Matrix<float>> _hiddenLayers, 
                                    out Matrix<float> _outputLayer, out List<Matrix<float>> _weights, 
                                    out List<Matrix<float>> _biases, out float _fitness, out ActivationType _activationType)
        {
            _inputLayer = data.inputLayer;
            _hiddenLayers = data.hiddenLayers;
            _outputLayer = data.outputLayer;
            _weights = data.weights;
            _biases = data.biases;
            _fitness = data.fitness;
            _activationType = data.activationType;
        }
        
        /// <summary>
        /// Calculates the data of the neural network and returns it as a TrainingDataStruct.
        /// </summary>
        /// <returns></returns>
        public TrainingDataStruct GetTrainingData()
        {
            CalculateData();
            return data;
        }

        /// <summary>
        /// Calculates the data of the neural network from the Layer arrays.
        /// </summary>
        private void CalculateData()
        {
            data.inputLayer = Matrix<float>.Build.DenseOfArray(new float[1, data.inputLayerArray.GetData().Length]);
            for (int i = 0; i < data.inputLayerArray.GetData().Length; i++)
            {
                data.inputLayer[0, i] = data.inputLayerArray.GetData()[i];
            }
            
            data.hiddenLayers = new List<Matrix<float>>();
            // loop through each hidden layer in the array
            foreach (Layer hiddenLayer in data.hiddenLayerArray)
            {
                // get the data from the hidden layer
                float[] hiddenLayerData = hiddenLayer.GetData();
                Matrix<float> hiddenLayerMatrix = null;
                
                // if this is the first hidden layer, create a matrix with the input layer as the row count
                {
                    hiddenLayerMatrix = Matrix<float>.Build.DenseOfArray(new float[1, hiddenLayerData.Length]);
                }

                for (int i = 0; i < hiddenLayerData.Length; i++)
                {
                    // set the value of the matrix to the value of the neuron
                    hiddenLayerMatrix[0, i] = hiddenLayerData[i];
                }

                data.hiddenLayers.Add(hiddenLayerMatrix);
            }
            
            data.outputLayer = Matrix<float>.Build.DenseOfArray(new float[1, data.outputLayerArray.GetData().Length]);
            for (int i = 0; i < data.outputLayerArray.GetData().Length; i++)
            {
                data.outputLayer[0, i] = data.outputLayerArray.GetData()[i];
            }
            
            data.weights = new List<Matrix<float>>();
            foreach (Layer weight in data.weightsLayerArray)
            {
                float[] layerData = weight.GetData();
                Vector2Int layerDimensions = weight.GetDimensions();
                Matrix<float> weightMatrix = Matrix<float>.Build.DenseOfArray(new float[layerDimensions.x, layerDimensions.y]);
                // nested loop to loop through each row and column of the matrix
                for (int i = 0; i < layerDimensions.y; i++)
                {
                    for (int j = 0; j < layerDimensions.x; j++)
                    {
                        // set the value of the matrix to the value of the neuron
                        weightMatrix[j, i] = layerData[i * layerDimensions.x + j];
                    }
                }
                data.weights.Add(weightMatrix);
            }

            data.biases = new List<Matrix<float>>();
            foreach (Layer bias in data.biasesLayerArray)
            {
                float[] layerData = bias.GetData();
                Matrix<float> biasMatrix = Matrix<float>.Build.DenseOfArray(new float[1, layerData.Length]);
                for (int i = 0; i < layerData.Length; i++)
                {
                    biasMatrix[0, i] = layerData[i];
                }
                data.biases.Add(biasMatrix);
            }
        }

        /// <summary>
        /// Returns a bool to check if the training data is empty by using the fitness value.
        /// </summary>
        /// <returns></returns>
        public bool IsTrainingDataEmpty()
        {
            return data.fitness == 0;
        }
        
        /// <summary>
        /// Returns the file name of the training data.
        /// </summary>
        /// <returns></returns>
        public string GetFileName()
        {
            return fileName;
        }

        /// <summary>
        /// Sets the file name of the training data.
        /// </summary>
        /// <param name="_fileName"></param>
        public void SetFileName(string _fileName)
        {
            fileName = _fileName;
        }

        /// <summary>
        /// Returns the fitness of the training data.
        /// </summary>
        /// <returns></returns>
        public float GetFitness()
        {
            return data.fitness;
        }
        
        /// <summary>
        /// Clears the training data in the TrainingDataStruct
        /// </summary>
        public void ResetTrainingData()
        {
            // clear all data
            data.ClearData();
            ResetTrainingModel();
        }
        
        /// <summary>
        /// Sets the neural network data of the training data.
        /// Updates the training model struct.
        /// Saves the training model as a JSON file.
        /// </summary>
        public void SaveTrainingData()
        {
            SetNeuralNetworkData(agent.GetNeuralNetData());
            trainingModel = new TrainingModel(nnData, data);
            SaveLoadManager.Instance.Save<TrainingModel>(trainingModel.GetTrainingModel(), fileName);
        }
        
        /// <summary>
        /// Loads the training model from a JSON file and converts it to training data for use in the neural network.
        /// </summary>
        public void LoadTrainingData()
        {
            //ResetTrainingData();
            SaveLoadManager.Instance.Load(ref trainingModel, fileName);
            ConvertTrainingModelToTrainingData();
        }

        /// <summary>
        /// Converts the training model to training data for use in the neural network.
        /// </summary>
        private void ConvertTrainingModelToTrainingData()
        {
            // set the data of the training data struct to the values of the training model
            data.fitness = trainingModel.fitness;
            data.activationType = trainingModel.activationType;
            
            data.inputLayerArray = new Layer();
            data.inputLayerArray.SetNeuronLength(trainingModel.inputLayerDimensions);
            
            data.hiddenLayers = new List<Matrix<float>>();
            data.hiddenLayerArray = new List<Layer>();
            foreach (float[] hiddenLayer in trainingModel.hiddenLayers)
            {
                // create the matrix for the hidden layer
                Matrix<float> hiddenLayerMatrix = Matrix<float>.Build.DenseOfArray(new float[1, hiddenLayer.Length]);
                // nested loop to loop through each row and column of the matrix
                for (int i = 0; i < hiddenLayer.Length; i++)
                {
                    // set the value of the matrix to the value of the neuron
                    hiddenLayerMatrix[0, i] = hiddenLayer[i];
                }
                Layer hiddenLayerArray = new Layer();
                hiddenLayerArray.SetNeuronLength(hiddenLayer.Length);
                data.hiddenLayerArray.Add(hiddenLayerArray);
            }
            
            data.weights = new List<Matrix<float>>();
            data.weightsLayerArray = new List<Layer>();
            for(int i = 0; i < trainingModel.weights.Count; i++)
            {
                Matrix<float> weightMatrix = Matrix<float>.Build.DenseOfArray(trainingModel.weights[i]);
                Layer weightLayer = new Layer();
                weightLayer.SetData(weightMatrix);
                data.weightsLayerArray.Add(weightLayer);
            }
            
            data.biases = new List<Matrix<float>>();
            data.biasesLayerArray = new List<Layer>();
            for(int i = 0; i < trainingModel.biases.Count; i++)
            {
                Matrix<float> biasMatrix = Matrix<float>.Build.DenseOfArray(trainingModel.biases[i]);
                Layer biasLayer = new Layer();
                biasLayer.SetData(biasMatrix);
                data.biasesLayerArray.Add(biasLayer);
            }
        }

        public void ResetTrainingModel()
        {
            trainingModel.fitness = 0;
            trainingModel.activationType = ActivationType.Sigmoid;
            trainingModel.inputLayerDimensions = 0;
            trainingModel.outputLayerDimensions = 0;
            trainingModel.hiddenLayers = null;
            trainingModel.weights = null;
            trainingModel.biases = null;
        }

        /// <summary>
        /// Sets the neural network data parameter
        /// </summary>
        /// <param name="_neuralNetData"></param>
        public void SetNeuralNetworkData(NeuralNetData _neuralNetData)
        {
            nnData = _neuralNetData;
        }
        
        public void LoadFromFileName(string _fileName)
        {
            SetFileName(_fileName);
            LoadTrainingData();
        }
        
        public void Load()
        {
            LoadTrainingData();
        }
        
        /// <summary>
        /// A struct to store the training data of the neural network in a format that can be saved as a JSON file.
        /// </summary>
        [Serializable]
        public struct TrainingModel
        {
            public float fitness;
            public int inputLayerDimensions;
            public int outputLayerDimensions;
            public List<float[]> hiddenLayers;
            public List<float[,]> weights;
            public List<float[,]> biases;
            public ActivationType activationType;

            /// <summary>
            /// A constructor to set the values of the training model.
            /// </summary>
            /// <param name="_nnData"></param>
            /// <param name="_tDataStruct"></param>
            public TrainingModel(NeuralNetData _nnData, TrainingDataStruct _tDataStruct)
            {
                // set the local variables to the values of the neural network data
                fitness = _tDataStruct.fitness;
                activationType = _tDataStruct.activationType;
                inputLayerDimensions = _nnData.GetInputCount();
                outputLayerDimensions = _nnData.GetOutputCount();
                
                // create a new list of float arrays for the hidden layers
                hiddenLayers = new List<float[]>();
                // loop through each hidden layer in the neural network data
                foreach (Layer hiddenLayer in _tDataStruct.hiddenLayerArray)
                {
                    // add a new float array to the list
                    hiddenLayers.Add(hiddenLayer.GetData());
                }
                
                // create a new list of float arrays for the weights
                weights = new List<float[,]>();
                // loop through each weight layer in the neural network data
                foreach (Layer weight in _tDataStruct.weightsLayerArray)
                {
                    // get the dimensions of the weight layer
                    Vector2Int dimensions = weight.GetDimensions();
                    // create a new float array with the dimensions of the weight layer
                    float[,] weightArray = new float[dimensions.x, dimensions.y];
                    // nested loop to loop through each row and column of the matrix
                    for (int i = 0; i < dimensions.y; i++)
                    {
                        for (int j = 0; j < dimensions.x; j++)
                        {
                            // set the value of the matrix to the value of the neuron
                            weightArray[j, i] = weight.GetData()[i * dimensions.x + j];
                        }
                    }
                    // add the weight array to the list
                    weights.Add(weightArray);
                }
                
                // create a new list of float arrays for the biases
                biases = new List<float[,]>();
                // loop through each bias layer in the neural network data
                foreach (Layer bias in _tDataStruct.biasesLayerArray)
                {
                    // get the dimensions of the bias layer
                    int length = bias.GetData().Length;
                    // create a new float array with the dimensions of the bias layer
                    float[,] biasArray = new float[1, length];
                    // nested loop to loop through each row and column of the matrix
                    for (int i = 0; i < length; i++)
                    {
                        // set the value of the matrix to the value of the neuron
                        biasArray[0, i] = bias.GetData()[i];
                    }
                    // add the bias array to the list
                    biases.Add(biasArray);
                }
            }
            
            /// <summary>
            /// Returns the training model
            /// </summary>
            /// <returns></returns>
            public TrainingModel GetTrainingModel()
            {
                return this;
            }
        }
    }

    /// <summary>
    /// A container for the training data of the neural network.
    /// </summary>
    [Serializable]
    public struct TrainingDataStruct
    {
        /// <summary>
        /// A dense matrix for the input layer's floats for the neural network.
        /// </summary>
        public Matrix<float> inputLayer;
        
        /// <summary>
        /// A list of dense matrices for the hidden layers' floats for the neural network.
        /// </summary>
        public List<Matrix<float>> hiddenLayers;
        
        /// <summary>
        /// A dense matrix for the output layer's floats for the neural network.
        /// </summary>
        public Matrix<float> outputLayer;
        
        /// <summary>
        /// A list of dense matrices for the weights' floats for the neural network.
        /// </summary>
        public List<Matrix<float>> weights;
        
        /// <summary>
        /// A list of dense matrices for the biases' floats for the neural network.
        /// </summary>
        public List<Matrix<float>> biases;
        
        
        [SerializeField,
        Tooltip("The fitness of the neural network.")]
        public float fitness;
        
        [SerializeField,
        Tooltip("The activation type of the neural network.")]
        public ActivationType activationType;
        
        [SerializeField,
        Tooltip("A Layer object for the input layer.")]
        public Layer inputLayerArray;
        
        [SerializeField,
        Tooltip("A Layer object for the output layer.")]
        public Layer outputLayerArray;
        
        [SerializeField,
        Tooltip("A list of Layer objects for the hidden layers.")]
        public List<Layer> hiddenLayerArray;
        
        [SerializeField,
        Tooltip("A list of Layer objects for the weights.")]
        public List<Layer> weightsLayerArray;
        
        [SerializeField,
        Tooltip("A list of Layer objects for the biases.")]
        public List<Layer> biasesLayerArray;

        /// <summary>
        /// A list of integers for the number of neurons in each hidden layer.
        /// </summary>
        private List<int> hiddenNeuronCountList;
        
        /// <summary>
        /// Sets the data of the training data struct and converts the dense matrices to float arrays for storing as
        /// a Layer object.
        /// </summary>
        /// <param name="_inputLayer"></param>
        /// <param name="_hiddenLayers"></param>
        /// <param name="_outputLayer"></param>
        /// <param name="_weights"></param>
        /// <param name="_biases"></param>
        /// <param name="_fitness"></param>
        /// <param name="_activationType"></param>
        public void SetData(Matrix<float> _inputLayer, List<Matrix<float>> _hiddenLayers, 
                            Matrix<float> _outputLayer, List<Matrix<float>> _weights, List<Matrix<float>> _biases,
                            float _fitness, ActivationType _activationType)
        {
            inputLayer = _inputLayer;
            hiddenLayers = _hiddenLayers;
            outputLayer = _outputLayer;
            weights = _weights;
            biases = _biases;
            fitness = _fitness;
            activationType = _activationType;
            
            // convert from Matrix<float> to float[]
            inputLayerArray = new Layer();
            inputLayerArray.SetData(new float[inputLayer.RowCount * inputLayer.ColumnCount]);
            int index = 0;
            foreach (float input in inputLayer.Enumerate())
            {
                inputLayerArray.SetNeuron(index, input);
                index++;
            }

            outputLayerArray = new Layer();
            outputLayerArray.SetData(new float[outputLayer.RowCount * outputLayer.ColumnCount]);
            index = 0;
            foreach (float output in outputLayer.Enumerate())
            {
                outputLayerArray.SetNeuron(index, output);
                index++;
            }
            
            hiddenLayerArray = new List<Layer>();
            foreach (Matrix<float> hiddenLayer in hiddenLayers)
            {
                Layer layer = new Layer();
                layer.SetData(new float[hiddenLayer.RowCount * hiddenLayer.ColumnCount]);
                index = 0;
                foreach (float hiddenNeuron in hiddenLayer.Enumerate())
                {
                    layer.SetNeuron(index, hiddenNeuron);
                    index++;
                }
                hiddenLayerArray.Add(layer);
            }

            weightsLayerArray = new List<Layer>();
            foreach (Matrix<float> weight in weights)
            {
                Layer layer = new Layer();
                layer.SetData(weight);
                
                weightsLayerArray.Add(layer);
            }
            
            

            biasesLayerArray = new List<Layer>();
            foreach (Matrix<float> bias in biases)
            {
                Layer layer = new Layer();
                layer.SetData(new float[bias.RowCount * bias.ColumnCount]);
                index = 0;
                foreach (float biasValue in bias.Enumerate())
                {
                    layer.SetNeuron(index, biasValue);
                    index++;
                }
                biasesLayerArray.Add(layer);
            }

            hiddenNeuronCountList = new List<int>();
            foreach (Matrix<float> hiddenLayer in hiddenLayers)
            {
                hiddenNeuronCountList.Add(hiddenLayer.ColumnCount);
            }
        }

        /// <summary>
        /// Converts the dense matrices to float arrays for storing as a Layer object.
        /// </summary>
        public void ConvertData()
        {
            if(inputLayer != null)
            {
                if(inputLayerArray == null)
                {
                    // convert from Matrix<float> to float[]
                    inputLayerArray = new Layer();
                    inputLayerArray.SetData(new float[inputLayer.RowCount * inputLayer.ColumnCount]);
                    int index = 0;
                    foreach (float input in inputLayer.Enumerate())
                    {
                        inputLayerArray.SetNeuron(index, input);
                        index++;
                    }
                }
            }
            
            if(outputLayer != null)
            {
                if(outputLayerArray == null)
                {
                    outputLayerArray = new Layer();
                    outputLayerArray.SetData(new float[outputLayer.RowCount * outputLayer.ColumnCount]);
                    int index = 0;
                    foreach (float output in outputLayer.Enumerate())
                    {
                        outputLayerArray.SetNeuron(index, output);
                        index++;
                    }
                }
            }
        }

        /// <summary>
        /// Clears the data of the training data struct.
        /// </summary>
        public void ClearData()
        {
            inputLayer = null;
            hiddenLayers = null;
            outputLayer = null;
            weights = null;
            biases = null;
            fitness = 0;
            activationType = ActivationType.Sigmoid;
            inputLayerArray = null;
            outputLayerArray = null;
            hiddenLayerArray = null;
            weightsLayerArray = null;
            biasesLayerArray = null;
            hiddenNeuronCountList = null;
        }
        
        /// <summary>
        /// Returns a list of integers for the number of neurons in each hidden layer.
        /// </summary>
        /// <returns></returns>
        public List<int> GetHiddenNeuronCountList()
        { 
            return hiddenNeuronCountList;
        }
    }
}