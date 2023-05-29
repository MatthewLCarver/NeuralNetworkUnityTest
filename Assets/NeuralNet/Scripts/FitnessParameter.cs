using System;
using System.Collections.Generic;

using UnityEngine;

namespace NeuralNet
{
	[Serializable]
	public class FitnessParameter
	{
		[SerializeField]
		private string parameterName;
		[SerializeField]
		private float parameterValue;
		[SerializeField]
		private float parameterMultiplier;
		
		public FitnessParameter(string _parameterName, float _parameterValue, float _parameterMultiplier)
		{
			parameterName = _parameterName;
			parameterValue = _parameterValue;
			parameterMultiplier = _parameterMultiplier;
		}
		
		public string GetParameterName() => parameterName;

		public float GetParameterValue() => parameterValue;

		public float GetParameterMultiplier() => parameterMultiplier;

		public void SetParameterName(string _parameterName) => parameterName = _parameterName;

		public void SetParameterValue(float _parameterValue) => parameterValue = _parameterValue;

		public void SetParameterMultiplier(float _parameterMultiplier) => parameterMultiplier = _parameterMultiplier;

		public void ResetParameterValue()
		{
			parameterValue = 0f;
		}
	}
}