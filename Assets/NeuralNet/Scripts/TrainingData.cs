using System;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;

using Unity.VisualScripting;

using UnityEditor;
using UnityEngine.Rendering.Universal;

namespace NeuralNet
{
    [CreateAssetMenu(fileName = "TrainingData", menuName = "Neural Network/Training Data", order = 0)]
    public class TrainingData : ScriptableObject
    {
        [SerializeField]
        private TrainingDataStruct data;
        /*
        [SerializeField]
        private List<float> inputLayer;
        [SerializeField]
        private List<List<float>> hiddenLayers;
        [SerializeField]
        private List<float> outputLayer;
        [SerializeField]
        private List<List<float>> weights;
        [SerializeField]
        private List<List<float>> biases;*/


        public void SetTrainingData(Matrix<float> _inputLayer, List<Matrix<float>> _hiddenLayers, 
                                    Matrix<float> _outputLayer, List<Matrix<float>> _weights, List<Matrix<float>> _biases,
                                        ActivationType _activationType, float _learningRate = 0.1f)
        {
            data.SetData(_inputLayer, _hiddenLayers, _outputLayer, _weights, _biases, _learningRate, _activationType);
            /*data.inputLayer = _inputLayer;
            data.hiddenLayers = _hiddenLayers;
            data.outputLayer = _outputLayer;
            data.weights = _weights;
            data.biases = _biases;*/
        }
        
        public void SetTrainingData(TrainingDataStruct _data, float _fitness)
        {
            data.SetData(_data.inputLayer, _data.hiddenLayers, _data.outputLayer, _data.weights, _data.biases,
                _fitness, _data.activationType);

            //DisplayData();

        }

        /*private void DisplayData()
        {
            // convert from Matrix<float> to List<float>
            inputLayer = new List<float>();
            foreach (float input in data.inputLayer.Enumerate())
            {
                inputLayer.Add(input);
            }
            
            hiddenLayers = new List<List<float>>();
            foreach (Matrix<float> hiddenLayer in data.hiddenLayers)
            {
                List<float> hiddenLayerList = new List<float>();
                foreach (float hiddenNeuron in hiddenLayer.Enumerate())
                {
                    hiddenLayerList.Add(hiddenNeuron);
                }
                hiddenLayers.Add(hiddenLayerList);
            }
            
            outputLayer = new List<float>();
            foreach (float output in data.outputLayer.Enumerate())
            {
                outputLayer.Add(output);
            }
            
            weights = new List<List<float>>();
            foreach (Matrix<float> weight in data.weights)
            {
                List<float> weightList = new List<float>();
                foreach (float weightValue in weight.Enumerate())
                {
                    weightList.Add(weightValue);
                }
                weights.Add(weightList);
            }
            
            biases = new List<List<float>>();
            foreach (Matrix<float> bias in data.biases)
            {
                List<float> biasList = new List<float>();
                foreach (float biasValue in bias.Enumerate())
                {
                    biasList.Add(biasValue);
                }
                biases.Add(biasList);
            }
        }
        */

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
                
                /*// if this is the first hidden layer, create a matrix with the input layer as the row count
                if(index == 0)
                {
                    hiddenLayerMatrix = Matrix<float>.Build.DenseOfArray(new float[
                                                                             data.inputLayerArray.GetData().Length,
                                                                             hiddenLayerData.Length]);
                }
                else // otherwise, create a matrix with the previous hidden layer as the row count
                {
                    // get the previous hidden layer
                    hiddenLayerMatrix = Matrix<float>.Build.DenseOfArray(new float[
                                                                             data.hiddenLayers[index - 1].ColumnCount,
                                                                             hiddenLayerData.Length]);
                }*/
                for (int i = 0; i < hiddenLayerData.Length; i++)
                {
                    // set the value of the matrix to the value of the neuron
                    hiddenLayerMatrix[0, i] = hiddenLayerData[i];
                }
                
                /*for (int i = 0; i < hiddenLayerData.Length; i++)
                {
                    hiddenLayerMatrix[0, i] = hiddenLayerData[i];
                }*/

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

        public float GetFitness()
        {
            return data.fitness;
        }
        public void ResetTrainingData()
        {
            // clear all data
            data.ClearData();
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
                
                /*Layer layer = new Layer();
                layer.SetData(new float[weight.RowCount * weight.ColumnCount]);
                index = 0;
                foreach (float weightValue in weight.Enumerate())
                {
                    layer.SetNeuron(index, weightValue);
                    index++;
                }*/
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
        /*public void SetData(Matrix<float> _inputLayer, List<Matrix<float>> _hiddenLayers, 
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
            inputLayerArray = new float[inputLayer.RowCount * inputLayer.ColumnCount];
            int index = 0;
            foreach (float input in inputLayer.Enumerate())
            {
                inputLayerArray[index] = input;
                index++;
            }
            
            outputLayerArray = new float[outputLayer.RowCount * outputLayer.ColumnCount];
            index = 0;
            foreach (float output in outputLayer.Enumerate())
            {
                outputLayerArray[index] = output;
                index++;
            }
            
            hiddenLayerArray = new List<float[]>();
            for (int i = 0; i < hiddenLayers.Count; i++)
            {
                float[] layer = new float[hiddenLayers[i].RowCount * hiddenLayers[i].ColumnCount];
                hiddenLayerArray.Add(layer);
            }
            
            for (int i = 0; i < hiddenLayers.Count; i++)
            {
                index = 0;
                foreach (float hiddenNeuron in hiddenLayers[i].Enumerate())
                {
                    hiddenLayerArray[i][index] = hiddenNeuron;
                    index++;
                }
            }
            
            weightsArray = new float[weights.Count];
            for (int i = 0; i < weights.Count; i++)
            {
                weightsArray[i] = weights[i][0, 0];
            }
            
            biasesArray = new float[biases.Count];
            for (int i = 0; i < biases.Count; i++)
            {
                biasesArray[i] = biases[i][0, 0];
            }
        }*/

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

        /*public void Setup(NeuralNetData _nnData)
        {
            // THIS IS WHERE JSON DATA WILL BE READ IN
            // convert from float[] to Matrix<float>
            inputLayer = Matrix<float>.Build.Dense(1, inputLayerArray.Length, inputLayerArray);
            outputLayer = Matrix<float>.Build.Dense(1, outputLayerArray.Length, outputLayerArray);
            
            hiddenLayers = new List<Matrix<float>>();


            weights = new List<Matrix<float>>();
            for (int i = 0; i < weightsArray.Length; i++)
            {
                weights.Add(Matrix<float>.Build.Dense(1, 1, weightsArray[i]));
            }
            
            biases = new List<Matrix<float>>();

            for(int i = 0; i < biasesArray.Length; i++)
            {
                biases.Add(Matrix<float>.Build.Dense(1, 1, biasesArray[i]));
            }
        }*/
        public List<int> GetHiddenNeuronCountList()
        { 
            return hiddenNeuronCountList;
        }
    }
}