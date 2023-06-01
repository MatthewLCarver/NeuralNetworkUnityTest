using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class TrainingUI : MonoBehaviour
{
    [SerializeField,
    Tooltip("The time scale that is set when the button is clicked. Do not exceed your computer's processing power.")]
    private float timeScale = 1f;
    
    /// <summary>
    /// The text of the button that is used to display the time scale.
    /// </summary>
    private TMP_Text timeScaleButtonText;

    /// <summary>
    /// Sets the time scale button text to the time scale value.
    /// </summary>
    void Awake()
    {
        timeScaleButtonText = transform.GetChild(0).GetComponent<TMP_Text>();
        timeScaleButtonText.text = $"{timeScale}x";
    }
    
    /// <summary>
    /// Sets the time scale to the given value.
    /// </summary>
    public void TimeScaleButtonClicked()
    {
        Time.timeScale = timeScale;
    }
}
