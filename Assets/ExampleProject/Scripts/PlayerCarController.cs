using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerCarController : MonoBehaviour
{
    [SerializeField,
    Tooltip("The input vector that is used to move the car.")]
    private Vector3 input = Vector3.zero;
    
    /// <summary>
    /// The vertical input of the car.
    /// </summary>
    private float vert = 0;
    
    /// <summary>
    /// The horizontal input of the car.
    /// </summary>
    private float hor = 0;

    /// <summary>
    /// Listens for input and moves the car accordingly, applying drag to the car.
    /// </summary>
    void FixedUpdate()
    {
        if (Input.GetKey(KeyCode.W))
        {
            // set vert to lerp up to 1.0f
            vert = Mathf.Lerp(vert, 1.0f, 0.015f);
        }
        
        if (Input.GetKey(KeyCode.S))
        {
            // set vert to lerp down to -1.0f
            vert = Mathf.Lerp(vert, -1.0f, 0.02f);
        }
        
        if (Input.GetKey(KeyCode.A))
        {
            // set hor to lerp down to -1.0f
            hor = Mathf.Lerp(hor, -1.0f, 0.04f);
        }
        
        if (Input.GetKey(KeyCode.D))
        {
            // set hor to lerp up to 1.0f
            hor = Mathf.Lerp(hor, 1.0f, 0.04f);
        }

        // friction to 0 on both axis if no input is given
        if(vert != 0)
            vert = Mathf.Lerp(vert, 0.0f, 0.005f);
        if(hor != 0)
            hor = Mathf.Lerp(hor, 0.0f, 0.025f);
        
        MoveCar(vert, hor);
    }
    
    /// <summary>
    /// Moves the car according to the given input from the user
    /// </summary>
    /// <param name="_verticalMovement"></param>
    /// <param name="_horizontalMovement"></param>
    public void MoveCar(float _verticalMovement, float _horizontalMovement)
    {
        input = Vector3.Lerp(Vector3.zero, new Vector3(0, 0, _verticalMovement * 11.4f), 0.02f);
        input = transform.TransformDirection(input);
        transform.position += input;
        
        transform.eulerAngles += new Vector3(0, (_horizontalMovement * 90) * 0.02f, 0);
    }
}
