using SaveLoad;

using System;
using System.Collections;
using System.Collections.Generic;

using TMPro;

using UnityEngine;
using UnityEngine.UI;
using UnityEditor;
using UnityEngine.Networking;

public class MenuManager : MonoBehaviour
{
    [SerializeField]
    private TMP_Text debugText = null;

    private void FixedUpdate()
    {
        if(debugText == null)
            return;

        debugText.text = SaveLoadManager.Instance.GetSavePath();
    }

    /// <summary>
    /// Disables the given button.
    /// </summary>
    /// <param name="_button"></param>
    public void DisableButton(Button _button)
    {
        _button.interactable = false;
    }

    public void EnableButton(Button _button)
    {
        _button.interactable = true;
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
        UnityEngine.SceneManagement.SceneManager.LoadScene(currentSceneIndex - 1);
    }
}
