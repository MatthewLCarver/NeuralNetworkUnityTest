using SaveLoad;

using System;
using System.Collections;
using System.Collections.Generic;
using NeuralNet;
using TMPro;

using UnityEngine;
using UnityEngine.UI;

public class MenuManager : MonoBehaviour
{
    [SerializeField]
    private TMP_Text noTrainingModelText = null;
    
    [SerializeField] private bool isRacingScene = false;

    [SerializeField] private Button LoadModelButton; 
    [SerializeField] private Button PlayModelButton; 
    [SerializeField] private Button TrainModelButton;

    [SerializeField] private TrainingData trainingData;

    private void Start()
    {
        if(isRacingScene)
            StartCountdownTimer();
    }

    public void Restart()
    {
        //trainingData.ResetTrainingData();
        
        Application.LoadLevel(0);
        
        //SceneLoader.Instance.LoadPreviousScene();
    }

    /// <summary>
    /// Disables the given button.
    /// </summary>
    /// <param name="_button"></param>
    public void DisableButton(Button _button)
    {
        if(_button)
            _button.interactable = false;
    }

    public void EnableButton(Button _button)
    {
        if(_button)
            _button.interactable = true;
    }

    public void UpdateNoTrainingDataText()
    {
        noTrainingModelText.text = "No training data found. Please train a model first.";
        noTrainingModelText.fontSize = 75;
        StartCoroutine(ResetButtons());
    }

    private IEnumerator ResetButtons()
    {
        yield return new WaitForSeconds(.1f);
        DisableButton(PlayModelButton);
        EnableButton(LoadModelButton);
    }

    /// <summary>
    /// Goes to the next scene in the build order.
    /// </summary>
    public void GoToNextScene()
    {
        SceneLoader.Instance.LoadNextScene();
    }
    
    /// <summary>
    /// Goes to the previous scene in the build order.
    /// </summary>
    public void GoToPreviousScene()
    {
        SceneLoader.Instance.LoadPreviousScene();
    }
    
    public void GoToTrainingScene()
    {
        SceneLoader.Instance.LoadTrainingScene();
    }
    
    /// <summary>
    /// Quits the game in the editor and in the build.
    /// </summary>
    public void QuitGame()
    {
        #if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
        #endif
            Application.Quit();
    }

    public void ShowModelPath()
    {
        if (SaveLoadManager.Instance && SaveLoadManager.Instance.SaveFileExists())
        {
            noTrainingModelText.text = "Loaded from path:\n" + SaveLoadManager.Instance.GetSavePath();
            noTrainingModelText.fontSize = 30;
        }
    }

    public void StartCountdownTimer()
    {
        StartCoroutine(CountdownTimer());
    }

    private IEnumerator CountdownTimer()
    {
        noTrainingModelText.text = "3";
        yield return new WaitForSeconds(1f);
        
        noTrainingModelText.text = "2";
        yield return new WaitForSeconds(1f);
        
        noTrainingModelText.text = "1";
        yield return new WaitForSeconds(1);
        
        noTrainingModelText.text = "Race!";
        yield return new WaitForSeconds(1f);
        noTrainingModelText.text = "";
    }
}

public class SceneLoader
{
    /// <summary>
    /// Creates an instance of the SceneLoader.
    /// </summary>
    private static SceneLoader instance = new SceneLoader();
    
    /// <summary>
    /// The instance of the SceneLoader.
    /// </summary>
    public static SceneLoader Instance { get { return instance; } }
    
    /// <summary>
    /// Loads the next scene in the build order.
    /// </summary>
    public void LoadNextScene()
    {
        // get the current scene index
        int currentSceneIndex = UnityEngine.SceneManagement.SceneManager.GetActiveScene().buildIndex;
        // load the next scene
        UnityEngine.SceneManagement.SceneManager.LoadScene(currentSceneIndex + 1);
    }
    
    /// <summary>
    /// Loads the previous scene in the build order.
    /// </summary>
    public void LoadPreviousScene()
    {
        // get the current scene index
        int currentSceneIndex = UnityEngine.SceneManagement.SceneManager.GetActiveScene().buildIndex;
        // load the previous scene
        if(currentSceneIndex > 0)
            UnityEngine.SceneManagement.SceneManager.LoadScene(currentSceneIndex - 1);
    }

    public void LoadTrainingScene()
    {
        UnityEngine.SceneManagement.SceneManager.LoadScene(2);
    }
}
