using NeuralNet;
using UnityEditor;
using UnityEngine;

/// <summary>
/// A custom editor for the training data scriptable object to add buttons to reset, save and load the
/// training data and edit the file name to save.
/// </summary>
[CustomEditor(typeof(TrainingData))]
public class TrainingDataEditor : Editor
{
    /// <summary>
    /// The on inspector GUI function to add buttons and a text field to the inspector.
    /// </summary>
    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();
        TrainingData trainingData = (TrainingData) target;
        if (GUILayout.Button("Reset Training Data"))
        {
            trainingData.ResetTrainingData();
        }
        
        // An editable text field to change the file name of the training data.
        trainingData.SetFileName(EditorGUILayout.TextField("File Name", trainingData.GetFileName()));

        if(GUILayout.Button("Save Training Data"))
        {
            trainingData.SaveTrainingData();
        }
        
        if(GUILayout.Button("Load Training Data"))
        {
            trainingData.LoadTrainingData();
        }
    }
}
