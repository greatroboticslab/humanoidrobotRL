using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class HumanoidRobot : Agent

{
	
	public float curTime = 0f;
	public float timeout = 5f;
	public float weightRange = 1.0f;
	public float biasRange = 1.0f;
	public float torqueMult = 3000f;
	public Director director;
	public float fitness = 0;
	
	public Vector3 startPos;
	public Quaternion startRot;
	public ReferenceSkeleton refSkeleton;
	private float[] torques;
	
	public GameObject chest;
	
	public static float TanH(float input) {
		float e = 2.71f;
		//return (Mathf.Pow(e,2*input)-1)/(Mathf.Pow(e,2*input)+1);
		float numerator = (Mathf.Pow(e,input)-Mathf.Pow(e,-input));
		float denominator = (Mathf.Pow(e,input)+Mathf.Pow(e,-input));
		
		if(denominator == 0) {
			if(numerator > 0) {
				return 1f;
			}
			return -1f;
		}
		if(float.IsPositiveInfinity(numerator)) {
			return 1f;
		}
		if(float.IsNegativeInfinity(numerator)) {
			return -1f;
		}
		if(float.IsPositiveInfinity(denominator)) {
			return 0f;
		}
		if(float.IsNegativeInfinity(denominator)) {
			return 0f;
		}
		
		
		float output = numerator/denominator;
		//Debug.Log("raw : " + input + ", denom: " + denominator + ", output: " + output);
		return output;
	}
	
	[System.Serializable]
	public class ReferenceSkeleton {
		
		public Transform pelvis;
		public Transform leftFemur;
		public Transform leftTibia;
		public Transform leftFoot;
		public Transform rightFemur;
		public Transform rightTibia;
		public Transform rightFoot;
		
		public Transform lowerSpine;
		public Transform midSpine;
		public Transform upperSpine;
		public Transform neck;
		public Transform head;
		
		public Transform leftShoulder;
		public Transform leftBicep;
		public Transform leftForearm;
		public Transform leftHand;
		public Transform rightShoulder;
		public Transform rightBicep;
		public Transform rightForearm;
		public Transform rightHand;
		
		
		
	};
	
	public class NNLayer {
		
		public float[][] weights;
		public float[] biases;
		
	};
	
	public class NeuralNetwork {
		public float[] input;
		public List<NNLayer> layers;
		
		public float[] SaveNetwork() {
			
			List<float> lData = new List<float>();
			foreach(NNLayer l in layers) {
				foreach(float[] f in l.weights) {
					foreach(float _f in f) {
						lData.Add(_f);
					}
				}
				foreach(float f in l.biases) {
					lData.Add(f);
				}
			}
			
			return lData.ToArray();
			
		}
		
		public void LoadNetwork(float[] data) {
			int idx = 0;
			for(int i = 0; i < layers.Count; i++) {
				for(int j = 0; j < layers[i].weights.Length; j++) {
					for(int k = 0; k < layers[i].weights[j].Length; k++) {
						layers[i].weights[j][k] = data[idx];
						idx += 1;
					}
				}
				for(int j = 0; j < layers[i].biases.Length; j++) {
					layers[i].biases[j] = data[idx];
					idx += 1;
				}
			}
		}
		
		public float[] ForwardPass() {
			
			float[] layerInput = input;
			
			float[] output = new float[layers[layers.Count-1].weights[0].Length];
			for(int i = 0; i < layers.Count; i++) {
				float[] layerOutput = new float[layers[i].weights[0].Length];
				for(int j = 0; j < layerOutput.Length; j++) {
					for(int k = 0; k < layers[i].weights.Length; k++) {
						layerOutput[j] += layers[i].weights[k][j] * layerInput[k];
						layerOutput[j] += layers[i].biases[j];
					}
				}
				layerInput = layerOutput;
			}
			output = layerInput;
			for(int i = 0; i < output.Length; i++) {
				output[i] = TanH(output[i]);
			}
			
			return output;
		}
		
		public float[] ForwardPass(float[] _input) {
			
			
			for(int i = 0; i < _input.Length; i++) {
				input[i] = _input[i];
			}
			
			return ForwardPass();
			
		}
		
	}
	
	[System.Serializable]
	public class NeuralNetworkBase {
		public List<int> hiddenLayers;
	};
	
	[System.Serializable]
	public class HumanoidJoint {
		public GameObject gameObject;
		public HumanoidJoint parent;
		public Vector3 startPos;
		public Quaternion startRot;
		public Rigidbody rb;
		public float strength;
	};
	
	public NeuralNetworkBase neuralNetwork;
	public NeuralNetwork nn;
	
	public HumanoidJoint[] joints;
	public Collider[] contacts;
	public GameObject[] accelerometers;
	
	public void GenerateNeuralNetwork(bool randomize = false) {
		nn = new NeuralNetwork();
		nn.layers = new List<NNLayer>();
		List<float> inputs = new List<float>();
		//Contacts First
		foreach(Collider c in contacts) {
			inputs.Add(0);
		}
		
		
		//Output Layer, pitch yaw roll
		int outputNodes = 0;
		//outputNodes = GetInputs().Length;
		
		
		foreach(HumanoidJoint j in joints) {
			outputNodes += 3;
		}
		
		//nn.output = new float[outputNodes];
		
		nn.input = GetInputs();
		for(int i = 0; i < neuralNetwork.hiddenLayers.Count; i ++) {
			NNLayer newLayer = new NNLayer();
			int prevNodes = nn.input.Length;
			if(i > 0) {
				//prevNodes = nn.layers[i].weights[0].Length;
				prevNodes = neuralNetwork.hiddenLayers[i-1];
			}
			newLayer.weights = new float[prevNodes][];
			for(int j = 0; j < prevNodes; j++) {
				newLayer.weights[j] = new float[neuralNetwork.hiddenLayers[i]];
				if(randomize) {
					for(int k = 0; k < neuralNetwork.hiddenLayers[i]; k++) {
						newLayer.weights[j][k] = Random.Range(-weightRange,weightRange);
					}
				}
			}
			
			newLayer.biases = new float[neuralNetwork.hiddenLayers[i]];
			if(randomize) {
				for(int j = 0; j < neuralNetwork.hiddenLayers[i]; j++) {
					newLayer.biases[j] = Random.Range(-biasRange,biasRange);
				}
			}
			nn.layers.Add(newLayer);
			
		}
		//Add output layer
		
		//Debug.Log(nn.layers);
		int _prevNodes = nn.input.Length;
		if(nn.layers.Count > 0) {
			_prevNodes = nn.layers[neuralNetwork.hiddenLayers.Count-1].weights[0].Length;
		}
		NNLayer _newLayer = new NNLayer();
		_newLayer.weights = new float[_prevNodes][];
		for(int j = 0; j < _prevNodes; j++) {
			_newLayer.weights[j] = new float[outputNodes];
			if(randomize) {
				for(int k = 0; k < outputNodes; k++) {
					_newLayer.weights[j][k] = Random.Range(-weightRange,weightRange);
				}
			}
		}
		
		_newLayer.biases = new float[outputNodes];
		if(randomize) {
			for(int j = 0; j < outputNodes; j++) {
				_newLayer.biases[j] = Random.Range(-biasRange,biasRange);
			}
		}
		nn.layers.Add(_newLayer);
		
		
	}
	
	public void GetAllJoints() {
		
		List<GameObject> a = new List<GameObject>();
		List<Collider> c = new List<Collider>();
		List<HumanoidJoint> j = new List<HumanoidJoint>();
		List<GameObject> objs = GetChildren(gameObject);
		foreach(GameObject o in objs) {
			
			if(o.GetComponent<RobotJoint>()) {
				
				if(o.GetComponent<RobotJoint>().accelerometer) {
					a.Add(o);
				}
				
				HumanoidJoint newJoint = new HumanoidJoint();
				
				foreach(HumanoidJoint jo in joints) {
					if(jo.gameObject.transform == o.transform.parent) {
						newJoint.parent = jo;
					}
				}
				
				newJoint.startPos = o.transform.localPosition;
				newJoint.startRot = o.transform.localRotation;
				newJoint.gameObject = o;
				newJoint.strength = o.GetComponent<RobotJoint>().muscleStrength;
				newJoint.rb = o.GetComponent<Rigidbody>();
				j.Add(newJoint);
			}
			else {
				Collider col = o.GetComponent<Collider>();
				if(col) {
					if(col.isTrigger) {
						c.Add(col);
					}
				}
			}
		}
		
		contacts = c.ToArray();
		joints = j.ToArray();
		accelerometers = a.ToArray();
		
	}
	
	public List<GameObject> GetChildren(GameObject o) {
		List<GameObject> output = new List<GameObject>();
		
		output.Add(o);
		
		foreach(Transform child in o.transform) {
			output.AddRange(GetChildren(child.gameObject));
		}
		
		return output;
	}
	
	public void RunJoints(float[] torques) {
		
		
		
		//Debug.Log(torques[0]);
		
		for(int i = 0; i < torques.Length; i+=3) {
			joints[i/3].rb.AddTorque(new Vector3(torques[i]*torqueMult,torques[i+1]*torqueMult,torques[i+2]*torqueMult));
		}
	}
	
	public float[] GetInputs() {
		List<float> lInputs = new List<float>();
		
		//Time
		lInputs.Add(curTime);
		
		//Velocity
		foreach(GameObject a in accelerometers) {
			lInputs.Add(a.GetComponent<Rigidbody>().velocity.x);
			lInputs.Add(a.GetComponent<Rigidbody>().velocity.y);
			lInputs.Add(a.GetComponent<Rigidbody>().velocity.z);
		}
		
		//Angular Velocity
		foreach(GameObject a in accelerometers) {
			lInputs.Add(a.GetComponent<Rigidbody>().angularVelocity.x);
			lInputs.Add(a.GetComponent<Rigidbody>().angularVelocity.y);
			lInputs.Add(a.GetComponent<Rigidbody>().angularVelocity.z);
		}
		
		//Joint orientations
		foreach(HumanoidJoint j in joints) {
			
			lInputs.Add(j.gameObject.transform.localRotation.x);
			lInputs.Add(j.gameObject.transform.localRotation.y);
			lInputs.Add(j.gameObject.transform.localRotation.z);
			lInputs.Add(j.gameObject.transform.localRotation.w);
			
		}
		
		//Contact points
		foreach(Collider c in contacts) {
			if(c.gameObject.GetComponent<ContactPoint>().touching > 0) {
				lInputs.Add(1f);
			}
			else {
				lInputs.Add(0f);
			}
		}
		return lInputs.ToArray();
	}
	
	
	public void ResetSelf() {
		fitness = 0;
		
		//Primer
		foreach(HumanoidJoint j in joints) {
			j.rb.velocity = Vector3.zero;
			j.rb.angularVelocity = Vector3.zero;
			//j.rb.enabled = false;
		}
		
		transform.position = startPos;
		transform.rotation = startRot;
		
		//End
		foreach(HumanoidJoint j in joints) {
			j.rb.velocity = Vector3.zero;
			j.rb.angularVelocity = Vector3.zero;
			j.gameObject.transform.localPosition = j.startPos;
			j.gameObject.transform.localRotation = j.startRot;
			//j.rb.enabled = true;
		}
	}
	
	//MLAGENTS STUFF
	
	public override void OnEpisodeBegin() {
		
		curTime = 0f;
		ResetSelf();
		
	}
	
	public override void CollectObservations(VectorSensor sensor) {
		
		foreach(float f in GetInputs()) {
			sensor.AddObservation(f);
		}
		
	}
	
	public override void OnActionReceived(ActionBuffers actions) {
		//Debug.Log(actions.ContinuousActions[0]);
		for(int i = 0; i < joints.Length*3; i+=3) {
			float _torque = torqueMult*joints[i/3].strength;
			joints[i/3].rb.AddRelativeTorque(new Vector3(actions.ContinuousActions[i]*_torque, actions.ContinuousActions[i+1]*_torque,actions.ContinuousActions[i+2]*_torque));
			if(joints[i/3].parent != null) {
				joints[i/3].parent.rb.AddRelativeTorque(new Vector3(actions.ContinuousActions[i]*_torque, actions.ContinuousActions[i+1]*_torque,actions.ContinuousActions[i+2]*_torque)*-1);
			}
			//joints[i/3].rb.AddTorque(torqueMult, torqueMult,torqueMult);
		}
	}
	
    // Start is called before the first frame update
    void Start()
    {
		director = transform.parent.gameObject.GetComponent<Director>();
		startPos = transform.position;
		startRot = transform.rotation;
        GetAllJoints();
		Debug.Log("Inputs count: " + GetInputs().Length);
		Debug.Log("Outputs count: " + joints.Length*3);
		GenerateNeuralNetwork(true);
		
    }
	
	void Update() {
		
		if(Mathf.Abs(transform.position.x) > 3000) {
			SetReward(-9999f);
			EndEpisode();
		}
		if(Mathf.Abs(transform.position.y) > 3000) {
			SetReward(-9999f);
			EndEpisode();
		}
		if(Mathf.Abs(transform.position.z) > 3000) {
			SetReward(-9999f);
			EndEpisode();
		}
		
		//float[] inputs = GetInputs();
		//torques = nn.ForwardPass(inputs);
		
		//TEST
		/*
		for(int i = 0; i < joints.Length*3; i+=3) {
			float t = torqueMult * Mathf.Sin(curTime*4);
			joints[i/3].rb.AddRelativeTorque(new Vector3(t,0,t));
		}
		*/
		
		fitness += director.FrameFitness(this, "walk");
		
		if(curTime > timeout) {
			SetReward(director.GetFitness(this, "walk") + fitness);
			EndEpisode();
		}
		
		
	}

    // Update is called once per frame
    void FixedUpdate()
    {
		//RunJoints(torques);
		//refSkeleton.leftFemur.rotation = director.refSkeleton.leftFemur.rotation;
		curTime += Time.deltaTime;
	
    }
}
