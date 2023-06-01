using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Rotator : MonoBehaviour
{
    
    
    // Update is called once per frame
    void Update()
    {
        // rotate the object around its local Y axis at 1 degree per second
        transform.Rotate(Vector3.up * (Time.deltaTime * 10));
    }
}
