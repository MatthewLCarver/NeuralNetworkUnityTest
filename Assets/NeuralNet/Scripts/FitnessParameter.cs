using System;
using System.Collections.Generic;

using UnityEngine;

namespace NeuralNet
{
	[Serializable]
	public class FitnessParameter
	{
		[SerializeField, 
		 Tooltip("The name of the fitness parameter that is used to access the parameter in the " +
		         "fitness function of the TrainableAgent.")]
		private string parameterName;
		[SerializeField, 
		 Tooltip("The value of the fitness parameter that is calculated in the user written code and is used in the " +
		         "fitness function of the TrainableAgent.")]
		private float parameterValue;
		[SerializeField,
		Tooltip("The multiplier of the fitness parameter that is set by the user in editor or in runtime that is " +
		        "used in the fitness function of the TrainableAgent.")]
		private float parameterMultiplier;
		
		/// <summary>
		/// Sets a fitness parameter to be used in the fitness function of the TrainableAgent.
		/// </summary>
		/// <param name="_parameterName"></param>
		/// <param name="_parameterValue"></param>
		/// <param name="_parameterMultiplier"></param>
		public FitnessParameter(string _parameterName, float _parameterValue, float _parameterMultiplier)
		{
			parameterName = _parameterName;
			parameterValue = _parameterValue;
			parameterMultiplier = _parameterMultiplier;
		}
		
		/// <summary>
		/// Returns the name of the fitness parameter.
		/// </summary>
		/// <returns></returns>
		public string GetParameterName() => parameterName;

		/// <summary>
		/// Returns the value of the fitness parameter that is used in the fitness function.
		/// </summary>
		/// <returns></returns>
		public float GetParameterValue() => parameterValue;

		/// <summary>
		/// Returns the multiplier of the fitness parameter that is used in the fitness function.
		/// </summary>
		/// <returns></returns>
		public float GetParameterMultiplier() => parameterMultiplier;

		/// <summary>
		/// Sets the name of the fitness parameter.
		/// </summary>
		/// <param name="_parameterName"></param>
		public void SetParameterName(string _parameterName) => parameterName = _parameterName;

		/// <summary>
		/// Sets the value of the fitness parameter that is used in the fitness function.
		/// </summary>
		/// <param name="_parameterValue"></param>
		public void SetParameterValue(float _parameterValue) => parameterValue = _parameterValue;

		/// <summary>
		/// Sets the multiplier of the fitness parameter that is used in the fitness function.
		/// </summary>
		/// <param name="_parameterMultiplier"></param>
		public void SetParameterMultiplier(float _parameterMultiplier) => parameterMultiplier = _parameterMultiplier;

		/// <summary>
		/// Resets the value of the fitness parameter to 0.
		/// </summary>
		public void ResetParameterValue()
		{
			parameterValue = 0f;
		}
	}
}