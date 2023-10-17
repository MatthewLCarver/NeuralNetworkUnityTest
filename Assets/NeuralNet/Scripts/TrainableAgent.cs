using System;
using System.Collections.Generic;
using System.Linq;

using UnityEngine;
using UnityEngine.Events;

namespace NeuralNet
{
	/// <summary>
	/// An enum that is used to determine the type of the sensor input of the TrainableAgent to be used in
	/// the neural network.
	/// </summary>
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
		[SerializeField, Tooltip("The type of the sensor input of the TrainableAgent to be used in the neural network.")]
		private SensorType sensorType;
		
		/// <summary>
		/// The type of the sensor input of the TrainableAgent to be used in the neural network in the previous frame.
		/// </summary>
		private SensorType previousSensorType;

		[SerializeField, 
		 Range(3, 1000), 
		 Tooltip("The number of sensors that are used in the neural network. " +
		         "This value is used to determine the number of input neurons in the neural network." +
		         "If the sensor type is 3D, this value is multiplied by itself to determine the number of input neurons.")]
		private int sensorCount;
		
		[SerializeField,
		 Tooltip("The distance that the sensor raycasts fire to that is used as the input of the " +
		                        "neural network.")] 
		private float sensorRange;
		
		[SerializeField, 
		 Range(0, 360),
		Tooltip("The angle range (in degrees) that the sensors fire in")] 
		private float sensorAngle;
		
		[SerializeField, 
		 Tooltip("The position offset that the sensors fire from")] 
		private Vector3 sensorOriginOffset;
		
		[SerializeField, 
		 Tooltip("The layer mask that the sensor raycasts collide with. Used for the goal system that" +
		                         "is not implemented yet.")] 
		private LayerMask sensorLayerMask;
		
		[Space(5), 
		 SerializeField, 
		 Tooltip("The time interval that the sensor raycasts fire")] 
		private float sensorUpdateInterval;
		
		/// <summary>
		/// The timer that is used to determine when the sensor raycasts fire.
		/// </summary>
		private float sensorUpdateTimer;


		[SerializeField, 
		 Tooltip("The array that stores the sensor values. This array is used as the input of the " +
		         "neural network.")]
		private float[] sensors;
		
		/// <summary>
		/// The Ray array that is used to fire the sensor raycasts.
		/// </summary>
		private Ray[] rays;
		
		
		[Space(10), 
		 Header("Hidden Layer Parameters"), 
		 SerializeField ,
		 Range(1, 50), 
		 Tooltip("The list of integers that determines the number of neurons in each hidden layer. " +
		         "This list is used to determine the number of hidden layers and the number of neurons in " +
		         "each hidden layer in the neural network.")]
		private List<int> hiddenNeuronList;
		
		
		[Space(10), 
		 Header("Output Parameters"), 
		 SerializeField ,
		 Range(1, 20),
		 Tooltip("The number of output neurons in the neural network. This value is used to determine the " +
		         "number of output neurons in the neural network.")]
		private int outputCount;
		
		[SerializeField,
		Tooltip("The float array that stores the output values of the neural network. This array is used " +
		        "by a user to determine actions of their AI")]
		private float[] output;

		[Header("Fitness"), 
		 SerializeField,
		Tooltip("The best fitness score that the current training session has produced")] 
		private float bestFitness;
		
		[SerializeField,
		Tooltip("The fitness score of the current epoch")] 
		private float currentFitness;
		
		[SerializeField,
		Tooltip("The minimum fitness value that with be recorded and used." +
		        "Operates as a minimum threshold for the AI")] 
		private float minimumFitnessTarget;
		
		[SerializeField, 
		 Tooltip("In Seconds")] 
		private float minimumTimeTarget;
		
		[SerializeField, 
		 Tooltip("The current time of the current epoch (in seconds)")] 
		private float currentTime;
		
		[SerializeField,
		Tooltip("A list of fitness parameters that are used to determine the fitness score of the AI.")]
		private List<FitnessParameter> fitnessParameters;
		

		[Space(10), 
		 Header("Neural Network Parameters"), 
		 SerializeField,
		Tooltip("The neuralNetData scriptable object that stores the base data of the neural network.")]
		private NeuralNetData neuralNetData;
		
		[SerializeField,
		Tooltip("The trainingData scriptable object that stores the training data of the neural network.")]
		private TrainingData trainingData;
		
		[SerializeField,
		Tooltip("The Neural Network that drives the calculations for the AI")] 
		private NeuralNetwork neuralNetwork;
		
		/// <summary>
		/// A Unity event that is invoked when the AI resets.
		/// </summary>
		public UnityEvent resetAgentEvent;

		/// <summary>
		/// A Unity event that is invoked when the AI has no training data when attempting to load
		/// </summary>
		public UnityEvent noTrainingDataEvent;

		/// <summary>
		/// A flag to save this AI's training data before reset
		/// </summary>
		private bool isToBeSaved = false;

		/// <summary>
		/// Initialises the TrainableAgent on Awake
		/// </summary>
		private void Awake()
		{
			Initialise();
		}
		
		/// <summary>
		/// On collision, the fitness score is reset and the AI is reset event is invoked
		/// </summary>
		/// <param name="other"></param>
		private void OnCollisionEnter(Collision other)
		{
			if (other.gameObject.CompareTag("Obstacle"))
			{
				if (isToBeSaved)
				{
					trainingData.SaveTrainingData();
					isToBeSaved = false;
				}

				currentFitness = 0f;
				ResetFitnessParameters();

				resetAgentEvent?.Invoke();
			}
		}

		/// <summary>
		/// Initialise the Input, Hidden and Output layers and the Neural Network
		/// </summary>
		public void Initialise()
		{
			InitialiseSensors();

			InitialiseHiddenLayers();

			InitialiseOutput();
			
			InitialiseNeuralNetwork();
		}

		/// <summary>
		/// Initialises the sensors and rays based on the sensor type
		/// </summary>
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
		
		/// <summary>
		/// Initialises the hidden layers based on the hiddenNeuronList
		/// </summary>
		private void InitialiseHiddenLayers()
		{
			neuralNetData.SetHiddenLayerListCount(hiddenNeuronList);
		}
		
		/// <summary>
		/// Initialises the output layer based on the outputCount
		/// </summary>
		private void InitialiseOutput()
		{
			output = new float[outputCount];
			
			neuralNetData.SetOutputCount(outputCount);
		}
		
		/// <summary>
		/// Initialises the neural network based on the neuralNetData and trainingData
		/// </summary>
		private void InitialiseNeuralNetwork()
		{
			if((int)neuralNetwork.activationType == -1)
				Debug.LogError("Neural Network is null! Please assign a neural network reference to the TrainableAgent.");
			
			if((int)neuralNetData.GetActivationType() == -1)
				Debug.LogError("Neural Network Data is null! Please assign a neural network data reference to the TrainableAgent.");
			
			if(trainingData.IsTrainingDataEmpty())
			{
				trainingData.ResetTrainingData();
				
				neuralNetwork.Initialise(neuralNetData);
				trainingData.SetTrainingData(neuralNetwork.GetTrainingData(), bestFitness, this);
			}
			else
				neuralNetwork.Initialise(trainingData.GetTrainingData(), neuralNetData.GetNeuralNetworkData().learningRate);
			
		}

		/// <summary>
		/// Accessor function to get the neural network to start training
		/// </summary>
		public void Train(bool isActive = true)
		{
			if(isActive)
				StartTraining();
			else
				StartTraining(isActive);
		}
		
		/// <summary>
		/// Commences the training of the neural network thread and sets the neural network to active
		/// </summary>
		private void StartTraining(bool isActive = true)
		{
			neuralNetwork.StartThreading();
			neuralNetwork.SetNetworkActive(isActive);
		}

		/// <summary>
		/// Accessor function to get the neural network to stop training
		/// </summary>
		public void CeaseTraining()
		{
			StopTraining();
		}
		
		/// <summary>
		/// Aborts the training of the neural network thread and sets the neural network to inactive
		/// </summary>
		private void StopTraining()
		{
			neuralNetwork.StopThreading();
			neuralNetwork.SetNetworkActive(false);
		}

		public void ResetTrainingData()
		{
			trainingData.ResetTrainingData();
		}

		/// <summary>
		/// Updates the sensors, hidden layers and output layers of the neural network
		/// Calculates the overall fitness of the AI and updates the neural network
		/// (Trainable Agent parameters can be updated in the inspector in editor mode or at runtime in play mode
		/// without crashing and updates the neural network)
		/// </summary>
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
		
		/// <summary>
		/// Updates the sensors based on the sensor type and sensor update interval
		/// </summary>
		/// <exception cref="ArgumentOutOfRangeException"></exception>
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

		/// <summary>
		/// Updates the sensor data if any of the parameters have changed
		/// </summary>
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
		
		/// <summary>
		/// Updates the sensors in a cone shape on a 2D axis
		/// </summary>
		private void UpdateCone2DSensors()
		{
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

		/// <summary>
		/// Updates the sensors in all directions on a 2D axis
		/// </summary>
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

		/// <summary>
		/// Updates the sensors in a cone shape in a 3D space
		/// </summary>
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
		
		/// <summary>
		/// Updates the sensors in all directions in a 3D space
		/// </summary>
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
		
		/// <summary>
		/// Updates the sensor data
		/// </summary>
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
		
		/// <summary>
		/// Updates the hidden layer parameters if any of the parameters have changed
		/// </summary>
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

		/// <summary>
		/// Updates the output parameters if any of the parameters have changed
		/// </summary>
		private void UpdateOutputParameters()
		{
			if (output == null)
				return;
				
			if(outputCount != output.Length)
			{
				StopTraining();
				
				InitialiseOutput();
				trainingData.ResetTrainingData();
				InitialiseNeuralNetwork();
				
				StartTraining();
			}
		}

		/// <summary>
		/// Updates the output
		/// </summary>
		private void UpdateOutput()
		{
			output = neuralNetwork.GetOutput();
		}
		
		/// <summary>
		/// Updates the neural network
		/// </summary>
		private void UpdateNeuralNetwork()
        {
         	neuralNetwork.SetInput(sensors);
        }
		
		/// <summary>
		/// Calculates the fitness of the agent based on the user determined fitness parameters
		/// </summary>
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
				trainingData.SetTrainingData(neuralNetwork.GetTrainingData(), bestFitness, this);
				isToBeSaved = true;
				
				return;
			}
			
			if (currentTime > minimumTimeTarget && currentFitness < minimumFitnessTarget || 
             			    currentFitness < 0)
            {
             	ResetFitnessParameters();
             	resetAgentEvent?.Invoke();
            }
		}

		/// <summary>
		/// Resets the fitness value of all the fitness parameters
		/// </summary>
		private void ResetFitnessParameters()
		{
			foreach(FitnessParameter parameter in fitnessParameters)
			{
				parameter.ResetParameterValue();
			}
		}
		
		/// <summary>
		/// Sets the value of a fitness parameter based on the parameter name
		/// </summary>
		/// <param name="_parameterName"></param>
		/// <param name="_parameterValue"></param>
		public void SetFitnessParameter(string _parameterName, float _parameterValue)
		{
			foreach(FitnessParameter t in fitnessParameters.Where(t => t.GetParameterName() == _parameterName))
			{
				t.SetParameterValue(_parameterValue);
				break;
			}
		}
		
		/// <summary>
		/// Returns the output generated by the neural network
		/// </summary>
		/// <returns></returns>
		public float[] GetOutput()
		{
			return output;
		}

		/// <summary>
		/// Returns the number of inputs the neural network has
		/// </summary>
		/// <returns></returns>
		public int GetInputs()
		{
			return sensorCount;
		}

		/// <summary>
		/// Returns the float array of the input sensors
		/// </summary>
		/// <returns></returns>
		public float[] GetInputSensorArray()
		{
			return sensors;
		}

		/// <summary>
		/// Increments the epoch of the neural network
		/// </summary>
		public void IncrementEpoch()
		{
			neuralNetwork.IncrementEpoch();
		}
		
		/// <summary>
		/// Sets the current time of the agent
		/// </summary>
		/// <param name="_currentTime"></param>
		public void SetCurrentTime(float _currentTime)
		{
			currentTime = _currentTime;
		}

		/// <summary>
		/// Returns the neural network data struct
		/// </summary>
		/// <returns></returns>
		public NeuralNetData GetNeuralNetData()
		{
			return neuralNetData;
		}

		public void Load()
		{
			if(trainingData.ConfirmLoadTrainingData())
				Initialise();
			else
				noTrainingDataEvent?.Invoke();
		}

		public void LoadModel(string _fileName)
		{
			trainingData.LoadFromFileName(_fileName);
		}
	}
}