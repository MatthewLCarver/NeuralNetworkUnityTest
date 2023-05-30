using System;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;

using System.IO;

using Unity.VisualScripting;

using UnityEditor;
using UnityEngine.Rendering.Universal;
using Newtonsoft.Json;

using SaveLoad;

namespace NeuralNet
{
    [CreateAssetMenu(fileName = "TrainingData", menuName = "Neural Network/Training Data", order = 0)]
    public class TrainingData : ScriptableObject
    {
        [SerializeField]
        private TrainingDataStruct data;
        private string fileName = "FileName";

        public void SetTrainingData(Matrix<float> _inputLayer, List<Matrix<float>> _hiddenLayers, 
                                    Matrix<float> _outputLayer, List<Matrix<float>> _weights, List<Matrix<float>> _biases,
                                    ActivationType _activationType, float _learningRate = 0.1f)
        {
            data.SetData(_inputLayer, _hiddenLayers, _outputLayer, _weights, _biases, _learningRate, _activationType);
        }
        
        public void SetTrainingData(TrainingDataStruct _data, float _fitness)
        {
            data.SetData(_data.inputLayer, _data.hiddenLayers, _data.outputLayer, _data.weights, _data.biases,
                _fitness, _data.activationType);
        }


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
        
        public TrainingDataStruct GetTrainingData()
        {
            CalculateData();
            return data;
        }

        private void CalculateData()
        {
            data.inputLayer = Matrix<float>.Build.DenseOfArray(new float[1, data.inputLayerArray.GetData().Length]);
            for (int i = 0; i < data.inputLayerArray.GetData().Length; i++)
            {
                data.inputLayer[0, i] = data.inputLayerArray.GetData()[i];
            }
            
            // Create a new matrix
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

        public bool IsTrainingDataEmpty()
        {
            return data.fitness == 0;
        }
        
        public string GetFileName()
        {
            return fileName;
        }

        public void SetFileName(string _fileName)
        {
            fileName = _fileName;
        }

        public float GetFitness()
        {
            return data.fitness;
        }
        public void ResetTrainingData()
        {
            // clear all data
            data.ClearData();
        }
        
        public void SaveTrainingData()
        {
            // use the SaveLoadManager to save the training data
            SaveLoadManager.Instance.Save<TrainingDataStruct>(data, fileName);
        }
        
        public void LoadTrainingData()
        {
            // use the SaveLoadManager to load the training data
            SaveLoadManager.Instance.Load(ref data, fileName);
        }
    }
    
    [CustomEditor(typeof(TrainingData))]
    public class TrainingDataEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            base.OnInspectorGUI();
            TrainingData trainingData = (TrainingData) target;
            if (GUILayout.Button("Reset Training Data"))
            {
                trainingData.ResetTrainingData();
            }
            
            // create a editable string text field for the file name
            trainingData.SetFileName(EditorGUILayout.TextField("File Name", trainingData.GetFileName()));
            //(EditorGUILayout.TextField("File Name", trainingData.GetFileName()));

            if(GUILayout.Button("Save Training Data"))
            {
                trainingData.SaveTrainingData();
            }
            
            if(GUILayout.Button("Load Training Data"))
            {
                trainingData.LoadTrainingData();
            }
            
        }
    }

    [Serializable]
    public struct TrainingDataStruct
    {
        public Matrix<float> inputLayer;
        public List<Matrix<float>> hiddenLayers;
        public Matrix<float> outputLayer;
        public List<Matrix<float>> weights;
        public List<Matrix<float>> biases;
        [SerializeField]
        public float fitness;
        [SerializeField]
        public ActivationType activationType;
        [SerializeField]
        public Layer inputLayerArray;
        [SerializeField]
        public Layer outputLayerArray;
        [SerializeField]
        public List<Layer> hiddenLayerArray;
        [SerializeField]
        public List<Layer> weightsLayerArray;
        [SerializeField]
        public List<Layer> biasesLayerArray;

        private List<int> hiddenNeuronCountList;
        
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


        public List<int> GetHiddenNeuronCountList()
        { 
            return hiddenNeuronCountList;
        }
    }
}