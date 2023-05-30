using System;
using System.Collections;
using System.Collections.Generic;
using NeuralNet;
using UnityEngine;

public class CarController : MonoBehaviour
{
    private Vector3 startPosition;
    private Vector3 startRotation;

    [Range(-1f, 1f)] public float acceleration, turningValue;

    public float timeSinceStart = 0f;
    
    [Header("Fitness")]
    public float overallFitness;

    private Vector3 lastPosition;
    private float totalDistanceTravelled;
    private float avgSpeed;

    [Space(10), Header("Sensor Raycast Parameters")]
    [SerializeField] 
    private int sensorCount;
    private Ray[] rays;
    private float[] sensors;
    private float[] output;
    private Vector3 input;
    

    [SerializeField]
    private float trainingTimeInterval;
    private float time = 0f;
    
    private int inputTotal;
    private int outputTotal;
    private int hiddenLayerTotal;
    private List<int> hiddenNeuronTotalList;
    private ActivationType neuronActivationType;

    private TrainableAgent ta;
    
    private void Awake()
    {
        startPosition = transform.position;
        startRotation = transform.eulerAngles;
        
        ta = GetComponent<TrainableAgent>();

        sensorCount = ta.GetInputs();
        
        sensors = new float[sensorCount];
        
        ta.resetAgentEvent.AddListener(Reset);
        
        ta.Train();
    }

    private void OnDestroy()
    {
        ta.CeaseTraining();
    }

    private void OnApplicationQuit()
    {
        ta.CeaseTraining();
    }

    private void Update()
    {
        sensors = ta.GetInputSensorArray();
        
        time += Time.deltaTime;
        timeSinceStart += Time.deltaTime;
        ta.SetCurrentTime(timeSinceStart);
        output = ta.GetOutput();
        
        if (time >= trainingTimeInterval)
        {
            time = 0f;
            //output = neuralNetwork.RunNetwork(sensorArray);
            lastPosition = transform.position;

            if(output == null)
                return;
        
            acceleration = output[0];
            turningValue = output[1];
        
            MoveCar(acceleration, turningValue);
            CalculateDistance();
        }
    }

    public void Reset()
    {
        ta.CeaseTraining();
        
        timeSinceStart = 0f;
        totalDistanceTravelled = 0f;
        avgSpeed = 0f;
        lastPosition = startPosition;
        overallFitness = 0f;
        transform.position = startPosition;
        transform.eulerAngles = startRotation;
        
        ta.IncrementEpoch();
        
        // Test Code
        ta.Initialise();
        
        ta.Train();
    }

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

    public void MoveCar(float verticalMovement, float horizontalMovement)
    {
        input = Vector3.Lerp(Vector3.zero, new Vector3(0, 0, verticalMovement * 11.4f), 0.02f);
        input = transform.TransformDirection(input);
        transform.position += input;
        
        transform.eulerAngles += new Vector3(0, (horizontalMovement * 90) * 0.02f, 0);
    }
}
