using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class MenuManager : MonoBehaviour
{
    /// <summary>
    /// Disables the given button.
    /// </summary>
    /// <param name="_button"></param>
    public void DisableButton(Button _button)
    {
        _button.interactable = false;
    }
    
    public void GoToNextScene()
    {
        SceneLoader.Instance.LoadNextScene();
    }
    
    public void GoToPreviousScene()
    {
        SceneLoader.Instance.LoadPreviousScene();
    }
    
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
