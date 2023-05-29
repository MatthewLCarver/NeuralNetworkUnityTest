using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Random;
using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;

using Random = UnityEngine.Random;


namespace NeuralNet
{
    
    public enum ActivationType
    {
        Sigmoid,
        Tanh,
        ReLU,
        LeakyReLU,
        Softmax
    }

    public class NeuralNetwork : MonoBehaviour
    {
        public int inputCount;
        public int outputCount;
        public int hiddenLayerCount;
        public List<int> hiddenNeuronList;
        public ActivationType activationType;

        private List<TrainingDataStruct> trainingDataList = new List<TrainingDataStruct>();
    
        // 
        private Matrix<float> inputLayer;
    
        //
        private List<Matrix<float>> hiddenLayers;
    
        //
        private Matrix<float> outputLayer;
    
        //
        private List<Matrix<float>> weights;
    
        //
        private List<Matrix<float>> biases;
    
        //
        private float learningRate;
        
        private int totalLayers = 0;
        
        [SerializeField]
        private int totalEpochs = 0;
        
        private int currentEpoch = 0;
        
        [SerializeField]
        private int trainingExamples = 0;
        
        [SerializeField]
        private int miniBatchSize = 0;
        
        
        // Inputs and Outputs
        private float[] aiInput;
        private float[] aiOutput;
        
        // Multithreading
        private Thread networkThread;
        private bool networkActive;
        private float timeScale;

        /// <summary>
        /// Initialise the neural network with the data from the NeuralNetData scriptable object for a
        /// new neural network
        /// </summary>
        /// <param name="data"></param>
        /// <param name="_learningRate"></param>
        public void Initialise(NeuralNetData data, float _learningRate = 0.1f)
        {
            // Null propagation check
            inputLayer?.Clear();
            hiddenLayers?.Clear();
            outputLayer?.Clear();
            weights?.Clear();
            biases?.Clear();

            // Set the values
            inputCount = data.GetNeuralNetworkData().inputCount;
            outputCount = data.GetNeuralNetworkData().outputCount;
            hiddenLayerCount = data.GetNeuralNetworkData().hiddenLayerCount;
            hiddenNeuronList = data.GetNeuralNetworkData().hiddenNeuronList;
            activationType = data.GetNeuralNetworkData().activationType;
            learningRate = data.GetNeuralNetworkData().learningRate;
            totalLayers = hiddenLayerCount + 2;
            
            // Initialise the layers
            inputLayer = Matrix<float>.Build.Dense(1, inputCount);
            outputLayer = Matrix<float>.Build.Dense(1, outputCount);
            hiddenLayers = new List<Matrix<float>>();
            weights = new List<Matrix<float>>();
            biases = new List<Matrix<float>>();

            for (int i = 0; i < data.GetNeuralNetworkData().hiddenLayerCount; i++)
            {
                hiddenLayers.Add(Matrix<float>.Build.Dense(1, hiddenNeuronList[i]));
                
                // Weights
                if (i == 0)
                {
                    Matrix<float> inputToFirstHidden = 
                        Matrix<float>.Build.Dense(inputCount, hiddenNeuronList[i]);
                    weights.Add(inputToFirstHidden);
                }
                else
                {
                    Matrix<float> hiddenToHidden = 
                        Matrix<float>.Build.Dense(hiddenNeuronList[i - 1], 
                                                  hiddenNeuronList[i]);
                    weights.Add(hiddenToHidden);
                }
                
                // Biases
                biases.Add(Matrix<float>.Build.Dense(1, 
                                                     hiddenNeuronList[i]));
                biases[i] = biases[i].Map(x => Random.Range(-1f, 1f));
            }
            
            // Output Layer
            Matrix<float> outputWeight = 
                Matrix<float>.Build.Dense(hiddenNeuronList[hiddenLayerCount - 1], outputCount);
            weights.Add(outputWeight);
            biases.Add(Matrix<float>.Build.Dense(1, outputCount));
            biases[hiddenLayerCount] = biases[hiddenLayerCount].Map(x => Random.Range(-1f, 1f));
            
            // Initialise Weights
            RandomiseWeights();
            
            aiInput = new float[inputCount];
        }
    
        /// <summary>
        /// Initialise the neural network with the data from the TrainingDataStruct for an
        /// existing neural network training model
        /// </summary>
        /// <param name="data"></param>
        /// <param name="_learningRate"></param>
        public void Initialise(TrainingDataStruct data, float _learningRate = 0.1f)
        {
            // Null Propagation Check
            inputLayer?.Clear();
            hiddenLayers?.Clear();
            outputLayer?.Clear();
            weights?.Clear();
            biases?.Clear();


            // Set the local parameters
            inputCount = data.inputLayerArray.GetData().Length;
            outputCount = data.outputLayerArray.GetData().Length;
            hiddenNeuronList = data.GetHiddenNeuronCountList();

            activationType = data.activationType;
            learningRate = _learningRate;
            totalLayers = data.hiddenLayerArray.Count + 2;
            
            // Initialise the layers
            inputLayer = data.inputLayer;
            outputLayer = data.outputLayer;
            hiddenLayers = data.hiddenLayers;
            weights = data.weights;
            biases = data.biases;
            
            aiInput = new float[inputCount];
        }

        public void StartThreading()
        {
            networkThread = new Thread(RunNetwork);
            networkThread.Start();
        }

        public void StopThreading()
        {
            networkThread.Abort();
        }
        
        /// <summary>
        /// Loop through each neuron in the given layer and randomise the weights
        /// </summary>
        private void RandomiseWeights()
        {
            
            
            // Loop through each weight matrix and randomise the values
            for (int i = 0; i < weights.Count; i++)
            {
                // loop through each neuron in the layer
                for (int j = 0; j < weights[i].RowCount; j++)
                {
                    for (int k = 0; k < weights[i].ColumnCount; k++)
                    {
                        System.Random r = new System.Random();
                        float randomIndex = (float) (r.NextDecimal() * 2) - 1;
                        weights[i][j, k] = randomIndex;
                    }
                }
                /*weights[i] = weights[i].Map(x => randomIndex);*/
            }
        }

        /// <summary>
        /// Set the input data for the neural network
        /// </summary>
        /// <param name="_input"></param>
        public void SetInput(float[] _input)
        {
            aiInput ??= new float[_input.Length];
            aiInput = _input;
        }

        /// <summary>
        /// Get the output data from the neural network
        /// </summary>
        /// <returns></returns>
        public float[] GetOutput() => aiOutput;
        
        public void SetNetworkActive(bool _isActive) => networkActive = _isActive;



        /// <summary>
        /// Run the network with the given input data and return the output data as a float array
        /// </summary>
        /// <param name="_inputData"></param>
        /// <returns></returns>
        public void RunNetwork()
        {
            while (networkActive)
            {
                // initialise input layer with the raw data
                for (int i = 0; i < inputCount; i++)
                {
                    inputLayer[0, i] = aiInput[i];
                }

                // Forward Feed the data through the network
                ForwardFeed();

                // Convert the outputLayer to a float array to return
                if(outputLayer == null)
                    return;
                float[] output = new float[outputLayer.ColumnCount];
                for (int i = 0; i < outputLayer.ColumnCount; i++)
                    output[i] = outputLayer[0, i];

                // Mini-batch Gradient Descent
                //output = GradientDescent(output);
                if (currentEpoch >= totalEpochs)
                    output = MiniBatchGradientDescent(trainingExamples, ConvertTrainingDataList(trainingDataList),
                        miniBatchSize, learningRate, currentEpoch);

                aiOutput = output;
            }
        }

        /// <summary>
        /// Loops through each layer and activate the neurons whilst passing the data through the weights and biases
        /// </summary>
        private void ForwardFeed()
        {
            bool isComplete = false;
            for (int i = 0; i < totalLayers; i++)
            {
                ActivateNeurons(i, ref isComplete);
                if (isComplete)
                    break;
            }
        }

        private float[] TestRevisedMSEOutput()
        {
            // initialise input layer with the raw data
            for (int i = 0; i < inputCount; i++)
            {
                inputLayer[0, i] = aiInput[i];
            }

            // Forward Feed the data through the network
            ForwardFeed();
            
            // Convert the outputLayer to a float array to return
            if(outputLayer == null)
                return null;
            float[] output = new float[outputLayer.ColumnCount];
            for (int i = 0; i < outputLayer.ColumnCount; i++)
                output[i] = outputLayer[0, i];

            return output;
        }
        
        /// <summary>
        /// Runs the network with the given input data and returns the output data as a float array to be used for
        /// training the mini batch
        /// </summary>
        /// <param name="_miniBatch"></param>
        /// <returns></returns>
        private float[] TestMiniBatch(float[] _miniBatch)
        {
            // initialise input layer with the raw data
            for (int i = 0; i < inputCount; i++)
            {
                inputLayer[0, i] = _miniBatch[i];
            }
            
            // Forward Feed the data through the network
            // Loop through each layer and activate the neurons whilst passing the data through the weights and biases
            bool isComplete = false;
            for (int i = 0; i < totalLayers; i++)
            {
                ActivateNeurons(i, ref isComplete);
                if (isComplete)
                    break;
            }

            // Convert the outputLayer to a float array to return
            float[] output = new float[outputLayer.ColumnCount];
            for (int i = 0; i < outputLayer.ColumnCount; i++)
                output[i] = outputLayer[0, i];

            return output;
        }

        /// <summary>
        /// Conducts mini-batch gradient descent on the neural network with the given parameters and returns the output
        /// </summary>
        /// <param name="_trainingExamples"></param>
        /// <param name="_trainingData"></param>
        /// <param name="_miniBatchSize"></param>
        /// <param name="_learningRate"></param>
        /// <param name="_epochs"></param>
        /// <returns></returns>
        public float[] MiniBatchGradientDescent(int _trainingExamples, TrainingDataStruct[,] _trainingData, int _miniBatchSize, float _learningRate, int _epochs)
        {
            // Loop through each epoch
            for (int i = 0; i < _epochs; i++)
            {
                // randomly shuffle the training data
                ShuffleTrainingData(_trainingData);
                
                // Loop through each mini-batch
                for (int j = 0; j < _trainingExamples; j += _miniBatchSize)
                {
                    // Create a mini-batch
                    TrainingDataStruct[,] miniBatch = new TrainingDataStruct[_miniBatchSize, _trainingData.GetLength(1)];
                    for (int k = 0; k < _miniBatchSize; k++)
                    {
                        for (int l = 0; l < _trainingData.GetLength(1); l++)
                        {
                            miniBatch[k, l] = _trainingData[j + k, l];
                        }
                    }
                    
                    // Update the weights and biases
                    UpdateWeightsAndBiases(miniBatch, _learningRate);
                }
            }

            // Return the output layer
            float[] output = new float[outputLayer.ColumnCount];
            for (int i = 0; i < outputLayer.ColumnCount; i++)
                output[i] = outputLayer[0, i];
            return output;
        }

        /// <summary>
        /// Shuffle the training data to create a randomised mini-batch
        /// </summary>
        /// <param name="_trainingData"></param>
        private void ShuffleTrainingData(TrainingDataStruct[,] _trainingData)
        {
            // Loop through each training data
            for (int i = 0; i < _trainingData.GetLength(0); i++)
            {
                // Get a random index
                Thread.Sleep(1);
                System.Random r = new System.Random();
                int randomIndex = r.Next(0, _trainingData.GetLength(1) - 1);
                //int randomIndex = Random.Range(0, (int)index.Sample());

                // Swap the training data
                for (int j = 0; j < _trainingData.GetLength(1); j++)
                {
                    TrainingDataStruct temp = _trainingData[i, j];
                    _trainingData[i, j] = _trainingData[randomIndex, j];
                    _trainingData[randomIndex, j] = temp;
                }
            }
        }
        
        /// <summary>
        /// Update the weights and biases of the network using the mini-batch gradient descent algorithm
        /// </summary>
        /// <param name="_miniBatch"></param>
        /// <param name="_learningRate"></param>
        private void UpdateWeightsAndBiases(TrainingDataStruct[,] _miniBatch, float _learningRate)
        {

            // Loop through each training example in the mini-batch
            for(int i = 0; i < _miniBatch.GetLength(0); i++)
            {
                // Convert the data of the current minibatch
                _miniBatch[i, 0].ConvertData();
                
                // Get the desired outputs
                float[] desiredOutputs = new float[outputLayer.ColumnCount];
                for(int j = 0; j < outputLayer.ColumnCount; j++)
                {
                    desiredOutputs[j] = _miniBatch[i, 0].outputLayerArray.GetData()[j];
                }

                // Run the network with the input data
                float[] output = TestMiniBatch(_miniBatch[i, 0].inputLayerArray.GetData());
                float mse = GetMSECostOnly(output);

                // Calculate the error for each output neuron
                float[] errors = new float[outputLayer.ColumnCount];
                for(int j = 0; j < outputLayer.ColumnCount; j++)
                {
                    errors[j] = desiredOutputs[j] - output[j];
                }

                // Calculate the gradient for each output neuron
                float[] gradients = new float[outputLayer.ColumnCount];
                for(int j = 0; j < outputLayer.ColumnCount; j++)
                {
                    gradients[j] = errors[j] * GetActivationMethod(outputLayer[0, j]);
                }

                // Calculate the deltas for each output neuron
                float[,] deltas = new float[outputLayer.ColumnCount, hiddenLayers[hiddenLayers.Count - 1].ColumnCount];
                for(int j = 0; j < outputLayer.ColumnCount; j++)
                {
                    for(int k = 0; k < hiddenLayers[hiddenLayers.Count - 1].ColumnCount; k++)
                    {
                        deltas[j, k] = gradients[j] * hiddenLayers[hiddenLayers.Count - 1][0, k];
                    }
                }

                // Update the weights and biases for the output layer
                for(int j = 0; j < outputLayer.ColumnCount; j++)
                {
                    // Update the biases
                    biases[biases.Count - 1][0, j] += gradients[j] * _learningRate;

                    // Update the weights
                    for(int k = 0; k < hiddenLayers[hiddenLayers.Count - 1].ColumnCount; k++)
                    {
                        weights[weights.Count - 1][k, j] += deltas[j, k] * _learningRate;
                    }
                }

                Matrix<float>[] hiddenLayerErrors = new Matrix<float>[hiddenLayers.Count];
                for(int j = 0; j < hiddenLayers.Count; j++)
                {
                    hiddenLayerErrors[j] = Matrix<float>.Build.Dense(1, hiddenLayers[j].ColumnCount);
                }
                
                // Calculate the error for each hidden layer neuron
                for(int j = 0; j < hiddenLayers.Count - 1; j++)
                {
                    for(int k = 0; k < hiddenLayers[j].ColumnCount; k++)
                    {
                        float error = 0;
                        if(j == hiddenLayers.Count - 1)
                        {
                            for(int l = 0; l < outputLayer.ColumnCount; l++)
                            {
                                error += deltas[l, k] * weights[weights.Count - 1][k, l];
                            }
                        }
                        else
                        {
                            for(int l = 0; l < hiddenLayers[j + 1].ColumnCount; l++)
                            {
                                error += hiddenLayerErrors[j + 1][0, l] * weights[j + 1][k, l];
                            }
                        }
                        hiddenLayerErrors[j][0, k] = error;
                    }
                }
                
                // Calculate the gradient for each hidden layer neuron
                Matrix<float>[] hiddenLayerGradients = new Matrix<float>[hiddenLayers.Count];
                for(int j = 0; j < hiddenLayers.Count; j++)
                {
                    hiddenLayerGradients[j] = Matrix<float>.Build.Dense(1, hiddenLayers[j].ColumnCount);
                }
                
                for(int j = 0; j < hiddenLayers.Count; j++)
                {
                    for(int k = 0; k < hiddenLayers[j].ColumnCount; k++)
                    {
                        hiddenLayerGradients[j][0, k] = hiddenLayerErrors[j][0, k] * GetActivationMethod(hiddenLayers[j][0, k]);
                    }
                }
                
                // Calculate the deltas for each hidden layer neuron
                Matrix<float>[] hiddenLayerDeltas = new Matrix<float>[hiddenLayers.Count];
                for(int j = 0; j < hiddenLayers.Count; j++)
                {
                    hiddenLayerDeltas[j] = Matrix<float>.Build.Dense(hiddenLayers[j].ColumnCount, hiddenLayers[j].ColumnCount);
                }
                
                for(int j = 0; j < hiddenLayers.Count; j++)
                {
                    for(int k = 0; k < hiddenLayers[j].ColumnCount; k++)
                    {
                        for(int l = 0; l < hiddenLayers[j].ColumnCount; l++)
                        {
                            hiddenLayerDeltas[j][k, l] = hiddenLayerGradients[j][0, k] * hiddenLayers[j][0, l];
                        }
                    }
                }
                
                // Update the weights and biases for the hidden layers
                for(int j = 0; j < hiddenLayers.Count; j++)
                {
                    for(int k = 0; k < hiddenLayers[j].ColumnCount; k++)
                    {
                        // Update the biases
                        biases[j][0, k] += hiddenLayerGradients[j][0, k] * _learningRate;

                        // Update the weights
                        for(int l = 0; l < hiddenLayers[j].ColumnCount; l++)
                        {
                            weights[j][l, k] += hiddenLayerDeltas[j][k, l] * _learningRate;
                        }
                    }
                }

                // Update the weights and biases for the output layer
                for(int j = 0; j < outputLayer.ColumnCount; j++)
                {
                    // Update the biases
                    biases[biases.Count - 1][0, j] += gradients[j] * _learningRate;

                    // Update the weights
                    for(int k = 0; k < hiddenLayers[hiddenLayers.Count - 1].ColumnCount; k++)
                    {
                        weights[weights.Count - 1][k, j] += deltas[j, k] * _learningRate;
                    }
                }

                output = TestRevisedMSEOutput();
                
                // Calculate the new MSE
                float newMSE = 0;
                if (output != null)
                    newMSE = GetMSECostOnly(output);
                else
                    RevertWeightsAndBiases(hiddenLayerGradients, hiddenLayerDeltas, _learningRate);

                // Check if the MSE has increased
                if(mse > newMSE)
                    // If it has, decrease the learning rate
                    _learningRate *= 0.5f;
                else
                {
                    // If it hasn't, increase the learning rate
                    _learningRate *= 1.05f;
                    RevertWeightsAndBiases(hiddenLayerGradients, hiddenLayerDeltas, _learningRate);
                }

                /*ConcurrentDictionary<int, float[,]> hiddenLayerErrors = new ConcurrentDictionary<int, float[,]>();
                Parallel.For(0, hiddenLayers.Count, index =>
                {
                    hiddenLayerErrors.TryAdd(index, new float[1, hiddenLayers[index].ColumnCount]);
                });

                // Calculate the error for each hidden layer neuron
                Parallel.For(0, hiddenLayers.Count - 1, index =>
                {
                    for (int j = 0; j < hiddenLayers[index].ColumnCount; j++)
                    {
                        float error = 0;
                        if (index == hiddenLayers.Count - 1)
                        {
                            for (int k = 0; k < outputLayer.ColumnCount; k++)
                            {
                                error += deltas[k, j] * weights[weights.Count - 1][j, k];
                            }
                        }
                        else
                        {
                            for (int k = 0; k < hiddenLayers[index + 1].ColumnCount; k++)
                            {
                                error += hiddenLayerErrors[index + 1][k, j] * weights[index + 1][j, k];
                            }
                        }
                        hiddenLayerErrors[index][0, j] = error;
                    }
                });
                
                // Calculate the gradient for each hidden layer neuron
                ConcurrentDictionary<int, float[,]> hiddenLayerGradients = new ConcurrentDictionary<int, float[,]>();
                for(int index = 0; index < hiddenLayers.Count; index++)
                    hiddenLayerGradients[index] = 
                        new float[hiddenLayers[index].RowCount, hiddenLayers[index].ColumnCount];
                
                // Calculate the gradient for each hidden layer neuron
                Parallel.For(0, hiddenLayers.Count - 1, index =>
                {
                    for (int j = 0; j < hiddenLayers[index].ColumnCount; j++)
                    {
                        hiddenLayerGradients[index][0, j] = hiddenLayerErrors[index][0, j] * 
                                                            GetActivationMethod(hiddenLayers[index][0, j]);
                    }
                });
                
                // Calculate the deltas for each hidden layer neuron
                ConcurrentDictionary<int, float[,]> hiddenLayerDeltas = new ConcurrentDictionary<int, float[,]>();
                for(int index = 0; index < hiddenLayers.Count; index++)
                    hiddenLayerDeltas[index] = 
                        new float[hiddenLayers[index].RowCount, hiddenLayers[index].ColumnCount];
                
                // Calculate the deltas for each hidden layer neuron
                Parallel.For(0, hiddenLayers.Count - 1, index =>
                {
                    for (int j = 0; j < hiddenLayers[index].ColumnCount; j++)
                    {
                        for (int k = 0; k < hiddenLayers[index].RowCount; k++)
                        {
                            hiddenLayerDeltas[index][k, j] = hiddenLayerGradients[index][k, j] * 
                                                                hiddenLayers[index][k, j];
                        }
                    }
                });
                
                // Update the weights and biases for each hidden layer
                Parallel.For(0, hiddenLayers.Count - 1, index =>
                {
                    for (int j = 0; j < hiddenLayers[index].ColumnCount; j++)
                    {
                        // Update the biases
                        biases[index][0, j] += hiddenLayerGradients[index][0, j] * _learningRate;

                        // Update the weights
                        for (int k = 0; k < hiddenLayers[index].RowCount; k++)
                        {
                            weights[index][k, j] += hiddenLayerDeltas[index][k, j] * _learningRate;
                        }
                    }
                });
                
                // Update the weights and biases for the output layer
                for (int j = 0; j < outputLayer.ColumnCount; j++)
                {
                    // Update the biases
                    biases[biases.Count - 1][0, j] += gradients[j] * _learningRate;

                    // Update the weights
                    for (int k = 0; k < hiddenLayers[hiddenLayers.Count - 1].ColumnCount; k++)
                    {
                        weights[weights.Count - 1][k, j] += deltas[j, k] * _learningRate;
                    }
                }
                
                // Calculate the new MSE
                float newMSE = GetMSECostOnly(output);
                
                // If the new MSE is greater than the old MSE, revert the changes
                if (newMSE > mse)
                {
                    Debug.Log($"Old MSE: {mse} | New MSE: {newMSE} | Reverting changes...");
                    
                    // Revert the weights and biases for the output layer
                    for (int j = 0; j < outputLayer.ColumnCount; j++)
                    {
                        // Revert the biases
                        biases[biases.Count - 1][0, j] -= gradients[j] * _learningRate;

                        // Revert the weights
                        for (int k = 0; k < hiddenLayers[hiddenLayers.Count - 1].ColumnCount; k++)
                        {
                            weights[weights.Count - 1][k, j] -= deltas[j, k] * _learningRate;
                        }
                    }
                    
                    // Revert the weights and biases for each hidden layer
                    for (int n = 0; n < hiddenLayers.Count; n++)
                    {
                        for (int j = 0; j < hiddenLayers[n].ColumnCount; j++)
                        {
                            // Revert the biases
                            biases[n][0, j] -= hiddenLayerGradients[n][0, j] * _learningRate;

                            // Revert the weights
                            for (int k = 0; k < hiddenLayers[n].RowCount; k++)
                            {
                                weights[n][k, j] -= hiddenLayerDeltas[n][k, j] * _learningRate;
                            }
                        }
                    }
                }*/
                Debug.Log($"Old MSE: {mse} | New MSE: {newMSE} | ");
            }
        }

        private void RevertWeightsAndBiases(Matrix<float>[] _hiddenLayerGradients, Matrix<float>[] _hiddenLayerDeltas, float _learningRate)
        {
            for(int j = 0; j < outputLayer.ColumnCount; j++)
            {
                for(int n = 0; n < hiddenLayers.Count; n++)
                {
                    // Revert the biases
                    biases[n][0, j] -= _hiddenLayerGradients[n][0, j] * _learningRate;

                    // Revert the weights
                    for(int k = 0; k < hiddenLayers[n].RowCount; k++)
                    {
                        weights[n][k, j] -= _hiddenLayerDeltas[n][k, j] * _learningRate;
                    }
                }
            }
        }

        /// <summary>
        /// Conducts a gradient descent on the network using the backpropagation algorithm
        /// </summary>
        /// <param name="_output"></param>
        /// <returns></returns>
        private float[] GradientDescent(float[] _output)
        {
            float mse = GetMSECostOnly(_output);

            // perform mini-batch gradient descent on the hidden layer neurons utilising the weights, biases and activation function
            for (int i = 0; i < hiddenLayers.Count; i++)
            {
                // loop through each column in the hidden layer matrix
                for (int j = 0; j < hiddenLayers[i].ColumnCount; j++)
                {
                    // loop through each row of the column
                    for (int k = 0; k < hiddenLayers[i].RowCount; k++)
                    {
                        // store the old weight value
                        float oldWeight = weights[i][k, j];
                        
                        // store the new weight value
                        float newWeight = oldWeight + (learningRate * oldWeight);
                        
                        // store the old bias value
                        float oldBias = biases[i][0, j];
                        
                        // store the new bias value
                        float newBias = oldBias + (learningRate * oldBias);
                        
                        // update the weight and bias values
                        weights[i][k, j] = newWeight;
                        biases[i][0, j] = newBias;
                        
                        // get the new output
                        //_output = ReRunNetwork(inputLayer.ToRowMajorArray());
                        
                        // get the new MSE
                        float newMSE = GetMSECostOnly(_output);
                        
                        // if the new MSE is lower than the old MSE, keep the new values
                        if (newMSE < mse)
                        {
                            mse = newMSE;
                            //Debug.Log($"Old Cost: {mse} | New Cost: {newMSE}");
                        }
                        // otherwise, revert back to the old values
                        else
                        {
                            weights[i][k, j] = oldWeight;
                            biases[i][0, j] = oldBias;
                        }
                    }
                }
            }
            return _output;
        }

        /// <summary>
        /// Returns the Mean Square Error of the output
        /// </summary>
        /// <param name="_output"></param>
        /// <returns></returns>
        private float GetMSECostOnly(float[] _output)
        {
            // Get the Mean Square Error of the output
            float mse = 0;
            for (int i = 0; i < _output.Length; i++)
            {
                mse += Mathf.Pow(_output[i] - 1, 2);
            }
            
            return mse;
        }
        
        /// <summary>
        /// Returns the Mean Square Error of the output as a float array
        /// </summary>
        /// <param name="_output"></param>
        /// <returns></returns>
        private float[] GetAllOutputsMSE(float[] _output)
        {
            // Get the Mean Square Error of the output
            float[] outputMSE = new float[_output.Length];
            for (int i = 0; i < _output.Length; i++)
            {
                outputMSE[i] = Mathf.Pow(_output[i] - 1, 2);
            }

            return outputMSE;
        }

        /// <summary>
        /// Returns the Mean Square Error of the output against the target
        /// </summary>
        /// <param name="_output"></param>
        /// <param name="_target"></param>
        /// <returns></returns>
        private float GetMSE(float[] _output, float[] _target)
        {
            // Get the Mean Square Error of the output
            float mse = 0;
            for (int i = 0; i < _output.Length; i++)
            {
                mse += Mathf.Pow(_output[i] - _target[i], 2);
            }
            return mse;
        }
    
        
        /// <summary>
        /// Activate the neurons in the current layer and pass the data through the weights and biases
        /// if not the input layer
        /// </summary>
        /// <param name="_currentLayer"></param>
        /// <param name="_isComplete"></param>
        private void ActivateNeurons(int _currentLayer, ref bool _isComplete)
        {
            int layerAdjustment = _currentLayer - 1;
            switch (_currentLayer)
            {
                // Final Layer
                case var _ when _currentLayer == totalLayers - 1:
                    for (int i = 0; i < outputLayer.ColumnCount; i++)
                    {
                        outputLayer[0, i] = 
                            GetActivationMethod(hiddenLayers[_currentLayer-2][0, i] * 
                                weights[layerAdjustment][0, i] + 
                                biases[layerAdjustment][0, i]);
                    }
                    _isComplete = true;
                    break;
                
                // First Layer
                case 0:
                    for (int i = 0; i < inputLayer.ColumnCount; i++)
                    {
                        inputLayer[0, i] = 
                            GetActivationMethod(inputLayer[0, i]);
                    }
                    break;
                
                // Second Layer
                case 1:
                    for (int i = 0; i < hiddenLayers[_currentLayer - 1].ColumnCount; i++)
                    {
                        hiddenLayers[layerAdjustment][0, i] = 
                            GetActivationMethod(((inputLayer * 
                                                  weights[layerAdjustment])[0, i] + 
                                                 biases[layerAdjustment][0, i]));
                    }
                    break;
                
                // Error Handling
                case < 0:
                    Debug.LogError("Current Layer is less than 0");
                    break;
                /*case var _ when _currentLayer == totalLayers:
                    Debug.LogError("Current Layer is greater than total layers");
                    break;*/
                
                // Middle Layers
                default:
                    for(int i = 0; i < hiddenLayers[_currentLayer - 1].ColumnCount; i++)
                    {
                        hiddenLayers[layerAdjustment][0, i] = 
                            GetActivationMethod(((hiddenLayers[layerAdjustment - 1] * 
                                                  weights[layerAdjustment])[0, i] + 
                                                 biases[layerAdjustment][0, i]));
                    }
                    break;
                    
            }
        }

        #region ActivationMethods

        /// <summary>
        /// Get the activation method based on the activation type and return the value
        /// </summary>
        /// <param name="_weightedNeuron"></param>
        /// <returns></returns>
        private float GetActivationMethod(float _weightedNeuron)
        {
            switch (activationType)
            {
                case ActivationType.Sigmoid:
                    return Sigmoid(_weightedNeuron);
                case ActivationType.Tanh:
                    return Tanh(_weightedNeuron);
                case ActivationType.ReLU:
                    return RectifiedLinearUnit(_weightedNeuron);
                case ActivationType.LeakyReLU:
                    return LeakyRectifiedLinearUnit(_weightedNeuron);
                case ActivationType.Softmax:
                    return Softmax(_weightedNeuron);
                default:
                    Debug.LogError("Not a valid Activation Type");
                    break;
            }
            return 0;
        }
        
        /// <summary>
        /// Return the sigmoid of the weighted neuron
        /// </summary>
        /// <param name="_weightedNeuron"></param>
        /// <returns></returns>
        private float Sigmoid(float _weightedNeuron)
        {
            return 1 / (1 + Mathf.Exp(-_weightedNeuron));
        }
    
        /// <summary>
        /// Return the tanh of the weighted neuron
        /// </summary>
        /// <param name="_weightedNeuron"></param>
        /// <returns></returns>
        private float Tanh(float _weightedNeuron)
        {
            return (float) Math.Tanh(_weightedNeuron);
        }
        
        /// <summary>
        /// Return the ReLU of the weighted neuron
        /// </summary>
        /// <param name="_weightedNeuron"></param>
        /// <returns></returns>
        private float RectifiedLinearUnit(float _weightedNeuron)
        {
            return Mathf.Max(0, _weightedNeuron);
        }
        
        /// <summary>
        /// Return the Leaky ReLU of the weighted neuron
        /// </summary>
        /// <param name="_weightedNeuron"></param>
        /// <returns></returns>
        private float LeakyRectifiedLinearUnit(float _weightedNeuron)
        {
            return Mathf.Max(0.01f * _weightedNeuron, _weightedNeuron);
        }
        
        /// <summary>
        /// Return the Softmax of the weighted neuron
        /// </summary>
        /// <param name="_weightedNeuron"></param>
        /// <returns></returns>
        private float Softmax(float _weightedNeuron)
        {
            return Mathf.Exp(_weightedNeuron) / Mathf.Exp(_weightedNeuron);
        }
        
    #endregion ActivationMethods
        /// <summary>
        /// Increment the current epoch and add the training data to the list
        /// </summary>
        public void IncrementEpoch()
        {
            currentEpoch++;
            if(currentEpoch > totalEpochs)
                ResetEpoch();
            trainingDataList.Add(GetTrainingData());
        }
        
        /// <summary>
        /// Reset the current epoch and clear the training data list
        /// </summary>
        public void ResetEpoch()
        {
            currentEpoch = 0;
            trainingDataList.Clear();
        }
        
        
        private bool ValidateData()
        {
            // Check if the input layer is null
            if (inputLayer == null)
            {
                Debug.LogError("Input Layer is null");
                return false;
            }
            
            // Check if the output layer is null
            if (outputLayer == null)
            {
                Debug.LogError("Output Layer is null");
                return false;
            }
            
            // Check if the hidden layers are null
            if (hiddenLayers == null)
            {
                Debug.LogError("Hidden Layers are null");
                return false;
            }
            
            // Check if the weights are null
            if (weights == null)
            {
                Debug.LogError("Weights are null");
                return false;
            }
            
            // Check if the biases are null
            if (biases == null)
            {
                Debug.LogError("Biases are null");
                return false;
            }
            
            // Check if the training data list is null
            if (trainingDataList == null)
            {
                Debug.LogError("Training Data List is null");
                return false;
            }

            return true;

        }
        
        /// <summary>
        /// Returns the training data list as a 2D array of training data structs from a list of training data structs
        /// </summary>
        /// <param name="_trainingDataList"></param>
        /// <returns></returns>
        private TrainingDataStruct[,] ConvertTrainingDataList(List<TrainingDataStruct> _trainingDataList)
        {
            TrainingDataStruct[,] trainingData = new TrainingDataStruct[_trainingDataList.Count, 1];
            for (int i = 0; i < _trainingDataList.Count; i++)
            {
                trainingData[i, 0] = _trainingDataList[i];
            }

            return trainingData;
        }
        
        /// <summary>
        /// Returns a training data struct with the current training data
        /// </summary>
        /// <returns></returns>
        public TrainingDataStruct GetTrainingData()
        {
            TrainingDataStruct trainingData = new TrainingDataStruct();
            trainingData.inputLayer = inputLayer;
            trainingData.hiddenLayers = hiddenLayers;
            trainingData.outputLayer = outputLayer;
            trainingData.weights = weights;
            trainingData.biases = biases;
            trainingData.activationType = activationType;

            return trainingData;
        }
        
        public void SetTimeScale(float _timeScale)
        {
            timeScale = _timeScale;
        }
    }
}