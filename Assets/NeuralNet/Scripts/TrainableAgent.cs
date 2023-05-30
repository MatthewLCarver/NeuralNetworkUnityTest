using System;
using System.Collections.Generic;
using System.Linq;

using Unity.VisualScripting;

using UnityEngine;
using UnityEngine.Events;
using UnityEngine.Serialization;

namespace NeuralNet
{
	public enum SensorType
	{
		Cone2D,
		AllDirections2D,
		Cone3D,
		AllDirections3D,
	}

	public class TrainableAgent : MonoBehaviour
	{
		[Header("Sensor (Input) Parameters")]
		[SerializeField]
		private SensorType sensorType;
		private SensorType previousSensorType;

		[SerializeField, Range(3, 1000)] private int sensorCount;
		[SerializeField] private float sensorRange;
		[SerializeField, Range(0, 360)] private float sensorAngle;
		[SerializeField] private Vector3 sensorOriginOffset;
		[SerializeField] private LayerMask sensorLayerMask;
		[Space(5), SerializeField] private float sensorUpdateInterval;
		private float sensorUpdateTimer;


		[SerializeField]
		private float[] sensors;
		private Ray[] rays;
		
		
		[Space(10), Header("Hidden Layer Parameters"), SerializeField ,Range(1, 50)]
		private List<int> hiddenNeuronList;
		
		
		[Space(10), Header("Output Parameters"), SerializeField ,Range(1, 20)]
		private int outputCount;
		[SerializeField]
		private float[] output;

		[Header("Fitness"), SerializeField] private float bestFitness;
		[SerializeField] private float currentFitness;
		[SerializeField] private float minimumFitnessTarget;
		[SerializeField, Tooltip("In Seconds")] private float minimumTimeTarget;
		[SerializeField] private float currentTime;
		[SerializeField] private List<FitnessParameter> fitnessParameters;
		

		[Space(10), Header("Neural Network Parameters"), SerializeField]
		private NeuralNetData neuralNetData;
		[SerializeField] private TrainingData trainingData;
		[SerializeField] private NeuralNetwork neuralNetwork;
		
		public UnityEvent resetAgentEvent;

		private void Awake()
		{
			Initialise();
		}
		
		private void OnCollisionEnter(Collision other)
		{
			currentFitness = 0f;
			ResetFitnessParameters();
			resetAgentEvent?.Invoke();
		}

		public void Initialise()
		{
			InitialiseSensors();

			InitialiseHiddenLayers();

			InitialiseOutput();
			
			InitialiseNeuralNetwork();
		}

		private void InitialiseSensors()
		{
			if(sensorType is SensorType.Cone2D or SensorType.AllDirections2D)
			{
				sensors = new float[sensorCount];
				rays = new Ray[sensorCount];
			}
			else if(sensorType is SensorType.Cone3D or SensorType.AllDirections3D)
			{
				sensors = new float[sensorCount * sensorCount];
				rays = new Ray[sensorCount * sensorCount];
			}
			previousSensorType = sensorType;
			
			neuralNetData.SetInputCount(sensorCount);
		}
		
		
		private void InitialiseHiddenLayers()
		{
			neuralNetData.SetHiddenLayerListCount(hiddenNeuronList);
		}


		private void InitialiseOutput()
		{
			output = new float[outputCount];
			
			neuralNetData.SetOutputCount(outputCount);
		}
		
		private void InitialiseNeuralNetwork()
		{
			if(neuralNetwork == null)
				Debug.LogError("Neural Network is null! Please assign a neural network reference to the TrainableAgent.");
			
			if(trainingData.IsTrainingDataEmpty())
			{
				trainingData.ResetTrainingData();
				neuralNetwork.Initialise(neuralNetData);
				trainingData.SetTrainingData(neuralNetwork.GetTrainingData(), bestFitness);
			}
			else
				neuralNetwork.Initialise(trainingData.GetTrainingData(), neuralNetData.GetNeuralNetworkData().learningRate);
			
		}

		public void Train()
		{
			StartTraining();
		}
		
		private void StartTraining()
		{
			neuralNetwork.StartThreading();
			neuralNetwork.SetNetworkActive(true);
		}

		public void CeaseTraining()
		{
			StopTraining();
		}
		
		private void StopTraining()
		{
			neuralNetwork.StopThreading();
			neuralNetwork.SetNetworkActive(false);
		}

		private void FixedUpdate()
		{
			// Update the sensors
			UpdateSensorParameters();
			UpdateSensors();
			
			// Update the Hidden Layer parameters
			UpdateHiddenLayerParameters();

			// Update the Output Layer parameters
			UpdateOutputParameters();
			UpdateOutput();

			CalculateOverallFitness();

			// Update the neural network
			UpdateNeuralNetwork();
		}
		

		private void UpdateSensors()
		{
			sensorUpdateTimer += Time.deltaTime;
			
			if(sensorUpdateTimer >= sensorUpdateInterval)
			{
				sensorUpdateTimer = 0f;
				
				switch(sensorType)
				{
					case SensorType.Cone2D:
						UpdateCone2DSensors();
						break;

					case SensorType.AllDirections2D:
						UpdateAllDirections2DSensors();
						break;

					case SensorType.Cone3D:
						UpdateCone3DSensors();
						break;

					case SensorType.AllDirections3D:
						UpdateAllDirections3DSensors();
						break;

					default:
						throw new ArgumentOutOfRangeException();
				}
				
				// Update the sensor data
				UpdateSensorData();
			}
		}

		private void UpdateSensorParameters()
		{
			if(sensorCount != sensors.Length || sensorCount != rays.Length ||
			   sensorType != previousSensorType)
			{
				StopTraining();
				
				InitialiseSensors();
				trainingData.ResetTrainingData();
				InitialiseNeuralNetwork();
				
				StartTraining();
			}
		}
		
		private void UpdateCone2DSensors()
		{
			// cone center ray always points forward (z-axis) from the agent and is always the middle ray in the array
			/*rays[sensorCount / 2].direction = transform.forward;
			rays[sensorCount / 2].origin = transform.position + new Vector3(0, sensorHeight, 0);*/
			
			// calculate the angle increment between each ray
			float angleIncrement = sensorAngle / (sensorCount - 1);
			float angle = -sensorAngle / 2f;
			
			// loop through all the rays and update their direction and origin
			for(int i = 0; i < sensorCount; i++)
			{
				rays[i].direction = Quaternion.AngleAxis(angle, transform.up) * transform.forward;
				rays[i].origin = transform.position + sensorOriginOffset;
				
				angle += angleIncrement;
			}
		}

		private void UpdateAllDirections2DSensors()
		{
			float angle = 0f;
			float angleIncrement = 360f / sensorCount;

			for(int i = 0; i < sensorCount; i++)
			{
				rays[i].direction = new Vector3(Mathf.Cos(angle * Mathf.Deg2Rad), 
				                                0,
				                                Mathf.Sin(angle * Mathf.Deg2Rad));
				rays[i].origin = transform.position + sensorOriginOffset;

				angle += angleIncrement;
			}
		}

		private void UpdateCone3DSensors()
		{
			// calculate the angle increment between each ray in the horizontal and vertical directions respectively 
			float angleIncrement = sensorAngle / (sensorCount - 1);
			float angle = -sensorAngle / 2f;
			
			float heightAngleIncrement = 180f / sensorCount;
			float heightAngle = -90f;
			
			// loop through all the rays and update their direction and origin
			for(int i = 0; i < sensorCount; i++)
			{
				for(int j = 0; j < sensorCount; j++)
				{
					rays[i * sensorCount + j].direction = Quaternion.AngleAxis(angle, transform.up) * Quaternion.AngleAxis(heightAngle, transform.right) * transform.forward;
					rays[i * sensorCount + j].origin = transform.position + sensorOriginOffset;
					
					heightAngle += heightAngleIncrement;
				}
				
				angle += angleIncrement;
			}
		}
		
		private void UpdateAllDirections3DSensors()
		{
			// calculate the angle increment between each ray
			float angleIncrement = 360f / sensorCount;
			float angle = 0f;
			
			// loop through all the rays and update their direction and origin
			for(int i = 0; i < sensorCount; i++)
			{
				float heightAngleIncrement = 180f / sensorCount;
				float heightAngle = -90f;
				
				for(int j = 0; j < sensorCount; j++)
				{
					rays[i * sensorCount + j].direction = new Vector3(Mathf.Cos(heightAngle * Mathf.Deg2Rad) * Mathf.Cos(angle * Mathf.Deg2Rad), 
					                                                  Mathf.Sin(heightAngle * Mathf.Deg2Rad), 
					                                                  Mathf.Cos(heightAngle * Mathf.Deg2Rad) * Mathf.Sin(angle * Mathf.Deg2Rad));
					rays[i * sensorCount + j].origin = transform.position + sensorOriginOffset;
					
					heightAngle += heightAngleIncrement;
				}
				
				angle += angleIncrement;
			}
		}
		
		private void UpdateSensorData()
		{
			if(rays.Length != sensors.Length)
				sensors = new float[rays.Length];
			
			for(int i = 0; i < rays.Length; i++)
			{
				if(Physics.Raycast(rays[i], out RaycastHit hit, Single.MaxValue, sensorLayerMask))
				{
					sensors[i] = hit.distance / sensorRange;
				}
				Debug.DrawLine(rays[i].origin, hit.point, Color.red);
			}
			
			neuralNetwork.SetInput(sensors);
		}
		
		
		
		
		private void UpdateHiddenLayerParameters()
		{
			List<int> hiddenLayers = trainingData.GetTrainingData().GetHiddenNeuronCountList();
			if(hiddenLayers.Count != hiddenNeuronList.Count)
			{
				StopTraining();
				
				InitialiseHiddenLayers();
				trainingData.ResetTrainingData();
				InitialiseNeuralNetwork();
				
				StartTraining();
			}
			else
			{
				for(int i = 0; i < hiddenNeuronList.Count; i++)
				{
					if(hiddenLayers[i] != hiddenNeuronList[i])
					{
						StopTraining();
						
						trainingData.ResetTrainingData();
						InitialiseHiddenLayers();
						InitialiseNeuralNetwork();
						
						StartTraining();
						
						break;
					}
				}
			}
		}

		private void UpdateOutputParameters()
		{
			if(outputCount != output.Length)
			{
				StopTraining();
				
				InitialiseOutput();
				trainingData.ResetTrainingData();
				InitialiseNeuralNetwork();
				
				StartTraining();
			}
		}

		private void UpdateOutput()
		{
			output = neuralNetwork.GetOutput();
		}
		
		
		private void UpdateNeuralNetwork()
        {
         	neuralNetwork.SetInput(sensors);
        }
		
		private void CalculateOverallFitness()
		{
			currentFitness = 0.0f;
			foreach(FitnessParameter parameter in fitnessParameters)
			{
				currentFitness += parameter.GetParameterValue() * parameter.GetParameterMultiplier();
			}

			

			if (currentFitness > bestFitness && currentFitness > minimumFitnessTarget)
			{
				if (!float.IsInfinity(currentFitness))
					bestFitness = currentFitness;
				trainingData.SetTrainingData(neuralNetwork.GetTrainingData(), bestFitness);

				return;
			}
			
			if (currentTime > minimumTimeTarget && currentFitness < minimumFitnessTarget || 
             			    currentFitness < 0)
            {
             	ResetFitnessParameters();
             	resetAgentEvent?.Invoke();
            }
		}

		private void ResetFitnessParameters()
		{
			foreach(FitnessParameter parameter in fitnessParameters)
			{
				parameter.ResetParameterValue();
			}
		}


		public void SetFitnessParameter(string _parameterName, float _parameterValue)
		{
			foreach(FitnessParameter t in fitnessParameters.Where(t => t.GetParameterName() == _parameterName))
			{
				t.SetParameterValue(_parameterValue);
				break;
			}
		}
		
		public float[] GetOutput()
		{
			return output;
		}

		public int GetInputs()
		{
			return sensorCount;
		}

		public float[] GetInputSensorArray()
		{
			return sensors;
		}

		public void IncrementEpoch()
		{
			neuralNetwork.IncrementEpoch();
		}
		
		public void SetCurrentTime(float _currentTime)
		{
			currentTime = _currentTime;
		}
	}
}