using System;
using System.Collections;
using System.Collections.Generic;
using NeuralNet;
using UnityEngine;

public class RaceCarController : MonoBehaviour
{
    /// <summary>
    /// The starting position of the car.
    /// </summary>
    private Vector3 startPosition;
    
    /// <summary>
    /// The starting rotation of the car.
    /// </summary>
    private Vector3 startRotation;

    [SerializeField,
    Tooltip("The speed multiplier of the car.")]
    private float speed = 1f;

    [SerializeField,
     Range(-1f, 1f),
     Tooltip("The acceleration value of the car.")]
    private float acceleration;
    
    [SerializeField,
     Range(-1f, 1f),
     Tooltip("The rotation value of the car.")]
    private float turningValue;

    [SerializeField,
     Tooltip("The time since the car started moving.")]
    private float timeSinceStart = 0f;

    /// <summary>
    /// The last position of the car as a Vector3 before the car moved.
    /// </summary>
    private Vector3 lastPosition;
    
    /// <summary>
    /// The total distance travelled by the car.
    /// </summary>
    private float totalDistanceTravelled;
    
    /// <summary>
    /// The average speed of the car.
    /// </summary>
    private float avgSpeed;

    [Space(10),
     Header("Sensor Raycast Parameters"),
     SerializeField,
     Tooltip("The number of sensors that the car has.")]
    private int sensorCount;
    
    /// <summary>
    /// The array of rays that the car uses to sense its environment.
    /// </summary>
    private Ray[] rays;
    
    /// <summary>
    /// The array of sensor values that the car calculates for use by the TrainableAgent.
    /// </summary>
    private float[] sensors;
    
    /// <summary>
    /// The array of output values that the car receives from the TrainableAgent to use for movement.
    /// </summary>
    private float[] output;
    
    /// <summary>
    /// The input vector that the car uses to move.
    /// </summary>
    private Vector3 input;
    

    [SerializeField,
    Tooltip("The time interval between each application of the output.")]
    private float trainingTimeInterval;
    
    /// <summary>
    /// The time since the last application of the output.
    /// </summary>
    private float time = 0f;

    /// <summary>
    /// The TrainableAgent that the car uses to train.
    /// </summary>
    private TrainableAgent ta;
    
    /// <summary>
    /// Sets the starting position and rotation of the car, gets a reference to the TrainableAgent,
    /// and sets the sensor count before training the car.
    /// </summary>
    private void Awake()
    {
        startPosition = transform.position;
        startRotation = transform.eulerAngles;
        
        ta = GetComponent<TrainableAgent>();

        sensorCount = ta.GetInputs();
        sensors = new float[sensorCount];

        ta.Train();
    }

    /// <summary>
    /// Accessor variable to disable this script
    /// </summary>
    public void DisableScript()
    {
        enabled = false;
    }
    /// <summary>
    /// Stops the car from training when the car is destroyed.
    /// </summary>
    private void OnDestroy()
    {
        Reset();
        ta.CeaseTraining();
    }
    
    /// <summary>
    /// Stops the car from training when the car is disabled.
    /// </summary>
    private void OnDisable()
    {
        Reset();
        ta.CeaseTraining();
    }

    /// <summary>
    /// Stops the car from training when the game is closed.
    /// </summary>
    private void OnApplicationQuit()
    {
        Reset();
        ta.CeaseTraining();
    }

    /// <summary>
    /// Get the sensor array from the TrainableAgent's input sensor array, and the output array to the
    /// TrainableAgent's output array.
    /// Count the time since the last application of the output, and apply the output to the car.
    /// Calculate the distance travelled by the car.
    /// </summary>
    private void Update()
    {
        sensors = ta.GetInputSensorArray();
        
        time += Time.deltaTime;
        timeSinceStart += Time.deltaTime;
        ta.SetCurrentTime(timeSinceStart);
        if(timeSinceStart > 1)
            output = ta.GetOutput();
        
        if (time >= trainingTimeInterval)
        {
            time = 0f;
            if(output == null)
                return;
            
            lastPosition = transform.position;

            acceleration = output[0];
            turningValue = output[1];
        
            MoveCar(acceleration, turningValue);
            CalculateDistance();
        }
    }

    /// <summary>
    /// Resets the car to its starting position and rotation, and resets the TrainableAgent.
    /// </summary>
    public void Reset()
    {
        ta.CeaseTraining();
        
        timeSinceStart = 0f;
        totalDistanceTravelled = 0f;
        avgSpeed = 0f;
        lastPosition = startPosition;
        transform.position = startPosition;
        transform.eulerAngles = startRotation;
        
        ta.Initialise();
        
        ta.Train();
    }

    /// <summary>
    /// Calculates the fitness parameters of the car for use by the TrainableAgent.
    /// </summary>
    private void CalculateDistance()
    {
        if (Vector3.Dot(transform.forward, (transform.position - lastPosition)) > 0)
        {
            // Going forward
            float distanceFromStart = Vector3.Distance(transform.position, startPosition);
            totalDistanceTravelled += (Vector3.Distance(transform.position, lastPosition) * distanceFromStart);
        }
        else
        {
            // Going backwards
            totalDistanceTravelled -= Vector3.Distance(transform.position, lastPosition);
        }
        avgSpeed = totalDistanceTravelled / timeSinceStart;
        
        if(!Single.IsNaN(totalDistanceTravelled))
            ta.SetFitnessParameter("Distance", totalDistanceTravelled);
        if(!Single.IsNaN(avgSpeed))
            ta.SetFitnessParameter("Speed", avgSpeed);
        float sensorData = CalculateSensorData();
        if(!Single.IsNaN(sensorData))
            ta.SetFitnessParameter("Spacing", sensorData);
        
    }

    /// <summary>
    /// Calculates the sensor data and returns it
    /// </summary>
    /// <returns></returns>
    private float CalculateSensorData()
    {
        float totalSensorData = 0f;
        for (int i = 0; i < sensorCount; i++)
        {
            totalSensorData += sensors[i];
        }

        return (totalSensorData / sensorCount);
    }

    /// <summary>
    /// Moves the car based on the output values received from the TrainableAgent.
    /// </summary>
    /// <param name="_verticalMovement"></param>
    /// <param name="_horizontalMovement"></param>
    public void MoveCar(float verticalMovement, float horizontalMovement)
    {
        input = Vector3.Lerp(Vector3.zero, new Vector3(0, 0, verticalMovement * speed ), 0.02f);// * 11.4f
        input = transform.TransformDirection(input);
        transform.position += input;
        
        transform.eulerAngles += new Vector3(0, (horizontalMovement * 90) * 0.02f, 0);
    }
}