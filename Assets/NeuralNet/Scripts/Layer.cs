using MathNet.Numerics.LinearAlgebra;

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
    public class Layer
    {
        [SerializeField, Tooltip("Do NOT change this value. It is used to determine the dimensions of the Layer.")]
        private Vector2Int dimensions;
        [SerializeField, Tooltip("The total number of neurons in the Layer.")]
        private float[] neurons;
        
        /// <summary>
        /// Initializes the Layer with no neurons.
        /// </summary>
        public Layer()
        {
            neurons = null;
        }
        
        /// <summary>
        /// Initializes the Layer with the given neurons as an array of floats.
        /// </summary>
        /// <param name="_neurons"></param>
        public Layer(float[] _neurons)
        {
            neurons = _neurons;
        }
        
        /// <summary>
        /// Sets the dimensions and the neurons of the Layer with the given dense matrix.
        /// </summary>
        /// <param name="_neurons"></param>
        public void SetData(Matrix<float> _neurons)
        {
            dimensions = new Vector2Int(_neurons.RowCount, _neurons.ColumnCount);
            
            neurons = new float[_neurons.RowCount * _neurons.ColumnCount];
            int index = 0;
            foreach (float neuron in _neurons.Enumerate())
            {
                neurons[index] = neuron;
                index++;
            }
        }

        /// <summary>
        /// Sets a specific neuron in the Layer.
        /// </summary>
        /// <param name="_index"></param>
        /// <param name="_data"></param>
        public void SetNeuron(int _index, float _data)
        {
            neurons[_index] = _data;
        }
        
        /// <summary>
        /// Sets the neurons of the Layer with the given array of floats.
        /// </summary>
        /// <param name="_neurons"></param>
        public void SetData(float[] _neurons)
        {
            neurons = _neurons;
        }
        
        /// <summary>
        /// Sets the size of the neurons in the Layer with the given length.
        /// </summary>
        /// <param name="_length"></param>
        public void SetNeuronLength(int _length)
        {
            neurons = new float[_length];
        }
        
        /// <summary>
        /// Clears the neuron data of the Layer.
        /// </summary>
        public void ClearData()
        {
            neurons = null;
        }
        
        /// <summary>
        /// Returns the neuron data of the Layer.
        /// </summary>
        /// <returns></returns>
        public float[] GetData()
        {
            return neurons;
        }

        /// <summary>
        /// Returns the dimensions of the Layer.
        /// </summary>
        /// <returns></returns>
        public Vector2Int GetDimensions()
        {
            return dimensions;
        }
    }
