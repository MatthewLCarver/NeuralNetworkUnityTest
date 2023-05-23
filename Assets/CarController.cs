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
    public float distanceMultiplier = 1.4f;
    public float avgSpeedMultiplier = 0.2f;
    public float sensorMultiplier = 1f;

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
        if (time >= trainingTimeInterval)
        {
            time = 0f;
            //output = neuralNetwork.RunNetwork(sensorArray);
        }
        
        output = ta.GetOutput();
    }

    private void FixedUpdate()
    {
        /*InputSensors();*/
        lastPosition = transform.position;
        
        /*// Neural Network Code Here
        for (int i = 0; i < sensorCount; i++)
        {
            sensorArray[i] = sensors[i];
        }*/

        if(output == null)
            return;
        
        acceleration = output[0];
        turningValue = output[1];
        
        MoveCar(acceleration, turningValue);
        timeSinceStart += Time.deltaTime;
        CalculateDistance();
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
    
    private void OnCollisionEnter(Collision other)
    {
        overallFitness -= 100f;
        Reset();
        /*if (other.gameObject.CompareTag("Wall"))
        {
            Death();
        }*/
    }

    private void Death()
    {
        throw new NotImplementedException();
    }

    private void CalculateDistance()
    {
        if (Vector3.Dot(transform.forward, (transform.position - lastPosition)) > 0)
        {
            // Going forward
            totalDistanceTravelled += Vector3.Distance(transform.position, lastPosition);
        }
        else
        {
            // Going backwards
            totalDistanceTravelled -= Vector3.Distance(transform.position, lastPosition);
        }

        avgSpeed = totalDistanceTravelled / timeSinceStart;

        overallFitness = (totalDistanceTravelled * distanceMultiplier) +
                         (avgSpeed * avgSpeedMultiplier) + 
                         CalculateSensorData();
        
        
        if(timeSinceStart > 20 && overallFitness < 750 || overallFitness < 0)
        {
            Reset();
        }

        if (overallFitness >= 750)
        {
            if (overallFitness > trainingData.GetFitness())
            {
                // Save the neural network
                trainingData.SetTrainingData(neuralNetwork.GetTrainingData(), overallFitness);
                if (overallFitness >= 100000000)
                {
                    Reset();
                }
            }
        }
        
        if(timeSinceStart > 3000)
            Reset();
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

        return (totalSensorData / sensorCount) * sensorMultiplier;
    }
// 
    private void InputSensors()
    {
        Transform t = transform;
        Vector3 left = t.forward - (t.right * 4); // - transform.right - transform.right;
        Vector3 right = t.forward + (t.right * 4);// + transform.right + transform.right;

        // create an array of rays for each sensor direction
        
        
        Ray r = new Ray(t.position, right);
        RaycastHit hit;
        
        // loop through the rays, checking if they hit anything
        for (int i = 0; i < sensorCount; i++)
        {
            // set the direction of the ray in a semi circle around the car from left to right
            if (rays != null)
            {
                rays[i].origin = transform.position;
                rays[i].direction = Vector3.Lerp(left, right, i / (float)(sensorCount - 1));
            }
            else
                rays[i] = new Ray(transform.position, Vector3.Lerp(left, right, i / (float)(sensorCount - 1)));

            if (Physics.Raycast(rays[i], out hit))
            {
                sensors[i] = hit.distance / 20;
                //print ("A: " + aSensor);
                Debug.DrawLine(rays[i].origin, hit.point, Color.red);
            }
        }
    }
    
    public void MoveCar(float verticalMovement, float horizontalMovement)
    {
        input = Vector3.Lerp(Vector3.zero, new Vector3(0, 0, verticalMovement * 11.4f), 0.02f);
        input = transform.TransformDirection(input);
        transform.position += input;
        
        transform.eulerAngles += new Vector3(0, (horizontalMovement * 90) * 0.02f, 0);
    }
}
