using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class RacingWinTracker : MonoBehaviour
{
    private int lapCounter = -1;
    private int lapTotal = 1;
    private bool canLap = true;
    
    [SerializeField]
    private string winText = "You Win!";
    
    [SerializeField]
    private TMP_Text winTextUI = null;

    private void OnTriggerEnter(Collider other)
    {
        if (canLap)
        {
            if (other.gameObject.layer == 7)
            {
                lapCounter++;
                Debug.Log("Lap Counter: " + lapCounter);
                StartCoroutine(LapTimer());
            }

            if (lapCounter >= lapTotal)
            {
                DisplayText();
                Time.timeScale = 0;
            }
        }

    }

    private IEnumerator LapTimer()
    {
        canLap = false;
        yield return new WaitForSeconds(30);
        canLap = true;
    }

    private void DisplayText()
    {
        winTextUI.text = winText;
    }
}
