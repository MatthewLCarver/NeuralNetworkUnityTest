using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class TrainingUI : MonoBehaviour
{
    [SerializeField]
    private float timeScale = 1f;
    
    private TMP_Text timeScaleButtonText;

    void Awake()
    {
        timeScaleButtonText = transform.GetChild(0).GetComponent<TMP_Text>();
        timeScaleButtonText.text = $"{timeScale}x";
    }
    
    public void TimeScaleButtonClicked()
    {
        Time.timeScale = timeScale;
    }
}
