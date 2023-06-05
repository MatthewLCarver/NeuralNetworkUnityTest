using System.Collections;
using System.Collections.Generic;
using NeuralNet;
using UnityEngine;

public class MenuCar : MonoBehaviour
{
    private TrainableAgent ta;
    
    void Awake()
    {
        ta = GetComponent<TrainableAgent>();
        ta.Train(false);
        StartCoroutine(AbortThread());
    }

    private IEnumerator AbortThread()
    {
        yield return new WaitForSeconds(0.1f);
        ta.CeaseTraining();
    }
}
