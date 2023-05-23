using MathNet.Numerics.LinearAlgebra;

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
    public class Layer
    {
        [SerializeField, Tooltip("Do NOT change this value. It is used to determine the dimensions of the layer.")]
        private Vector2Int dimensions;
        [SerializeField]
        private float[] neurons;
        
        public Layer()
        {
            neurons = null;
        }
        
        public Layer(float[] _neurons)
        {
            neurons = _neurons;
        }
        
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

        public void SetNeuron(int _index, float _data)
        {
            neurons[_index] = _data;
        }
        
        public void SetData(float[] _neurons)
        {
            neurons = _neurons;
        }
        
        public void ClearData()
        {
            neurons = null;
        }
        
        public float[] GetData()
        {
            return neurons;
        }

        public Vector2Int GetDimensions()
        {
            return dimensions;
        }
    }
