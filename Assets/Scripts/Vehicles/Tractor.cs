using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Tractor : MonoBehaviour
{
	
	public Transform steeringWheel;
	public float lastSteeringRot;
	private float steeringDelta = 0;
	private float steeringRotation = 0;
	public float steeringMultiplier = 0.1f;
	public List<float> gearTorques;
	public List<float> gearTopSpeeds;
	public float brakes = 0.0f;
	public int currentGear = 1;
	
	public TractorWheelControl rearLeftWheel;
	public TractorWheelControl rearRightWheel;
	public TractorWheelControl frontLeftWheel;
	public TractorWheelControl frontRightWheel;
	
    // Start is called before the first frame update
    void Start()
    {
        lastSteeringRot = steeringWheel.localRotation.eulerAngles.y;
    }

    // Update is called once per frame
    void Update()
    {
		float sd1 = 0;
		float sd2 = 0;
		float sd3 = 0;
		float csteering = steeringWheel.localRotation.eulerAngles.y;
		sd1 = csteering - (lastSteeringRot);
		sd2 = csteering - (lastSteeringRot - 360);
		sd3 = csteering - (lastSteeringRot + 360);
		steeringDelta = sd1;
		if(Mathf.Abs(sd2) < Mathf.Abs(sd1) && Mathf.Abs(sd2) < Mathf.Abs(sd3)) {
			steeringDelta = sd2;
		}
		if(Mathf.Abs(sd3) < Mathf.Abs(sd1) && Mathf.Abs(sd3) < Mathf.Abs(sd2)) {
			steeringDelta = sd3;
		}
		steeringRotation += steeringDelta;
		
        Debug.Log(steeringRotation);
		
		frontRightWheel.WheelCollider.steerAngle = steeringRotation*steeringMultiplier;
		frontLeftWheel.WheelCollider.steerAngle = steeringRotation*steeringMultiplier;
		
		rearRightWheel.WheelCollider.motorTorque = gearTorques[currentGear] * (1f - brakes);
		rearLeftWheel.WheelCollider.motorTorque = gearTorques[currentGear] * (1f - brakes);
		
		//frontRightWheel.WheelCollider.steerAngle = Random.Range(0,90);
		
		lastSteeringRot = steeringWheel.localRotation.eulerAngles.y;
		
    }
}
