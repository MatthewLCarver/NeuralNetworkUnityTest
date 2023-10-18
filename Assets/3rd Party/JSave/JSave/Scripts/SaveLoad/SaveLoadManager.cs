﻿using Newtonsoft.Json;

using System;
using System.Collections.Generic;
using System.IO;
using MathNet.Numerics.LinearAlgebra;

using NeuralNet;
using Unity.VisualScripting;
using UnityEngine;

namespace SaveLoad
{
	public class SaveLoadManager : MonoBehaviour
	{
		[SerializeField] private string defSaveName = "TrainedModel";
		
		private const string PREFAB_PATH = "Prefabs/";

		private const string SCRIPTABLE_PATH = "Data/";

		private Dictionary<string, GameObject> prefabs;

		private Dictionary<string, ScriptableObject> scriptables;

		private string SAVE_PATH => $"{Application.persistentDataPath}/{defSaveName}.json";

		public bool encrypt = true;
		
		public bool createDefaultFile = true;

		private static JsonSerializerSettings jsonSettings => new JsonSerializerSettings
		{
			TypeNameHandling = TypeNameHandling.None,
			Formatting = Formatting.Indented,
		};

		public static SaveLoadManager Instance;

		private void Start()
		{
			if(Instance == null)
			{
				Instance = this;
				DontDestroyOnLoad(gameObject);
			}
			else
			{
				Destroy(gameObject);
			}
			
			if(!SaveFileExists() && createDefaultFile)
			{
				SaveGame(SAVE_PATH);
			}
			
			/*#if UNITY_EDITOR
				SAVE_PATH = $"{Application.persistentDataPath}/{defSaveName}.json";
			#else
				SAVE_PATH = $"{}"
			#endif*/
			
			prefabs = LoadPrefabs(PREFAB_PATH);
			scriptables = LoadScriptables(SCRIPTABLE_PATH);
			
		}

		private void Update()
		{
			if(Input.GetKeyDown(KeyCode.Alpha7))
			{
				Save();
			}
			if(Input.GetKeyDown(KeyCode.Alpha8))
			{
				Load();
			}
		}

		private Dictionary<string, GameObject> LoadPrefabs(string path)
		{
			Dictionary<string, GameObject> prefabs = new Dictionary<string, GameObject>();

			GameObject[] prefabsArray = Resources.LoadAll<GameObject>(path);
			foreach(GameObject prefab in prefabsArray)
			{
				SaveableEntity saveableEntity = prefab.GetComponent<SaveableEntity>();
				if(saveableEntity != null)
				{
					prefabs.Add(saveableEntity.metaData.prefabGuid, prefab);
					Debug.Log($"Loaded prefab {prefab.name} with guid {saveableEntity.metaData.prefabGuid}");
				}
				else
				{
					Debug.LogWarning($"Prefab {prefab.name} is not a saveable entity");
				}
			}

			return prefabs;
		}

		private Dictionary<string, ScriptableObject> LoadScriptables(string path)
		{
			Dictionary<string, ScriptableObject> scriptables = new Dictionary<string, ScriptableObject>();

			ScriptableObject[] scriptablesArray = Resources.LoadAll<ScriptableObject>(path);
			foreach(ScriptableObject scriptable in scriptablesArray)
			{
				scriptables.Add(scriptable.name, scriptable);
				Debug.Log(scriptable.name);
			}

			return scriptables;
		}

		public GameObject GetPrefabByName(string name)
		{
			if(prefabs.ContainsKey(name))
			{
				return prefabs[name];
			}
			else
			{
				Debug.LogError($"Prefab {name} not found");
				return null;
			}
		}

		public ScriptableObject GetScriptableByName(string name)
		{
			if(scriptables.ContainsKey(name))
			{
				return scriptables[name];
			}
			else
			{
				Debug.LogError($"Scriptable {name} not found");
				return null;
			}
		}
		
		public SaveableEntity FindSaveableEntityByGuid(string guid)
		{
			SaveableEntity[] saveableEntities = FindObjectsOfType<SaveableEntity>();
			foreach(SaveableEntity saveableEntity in saveableEntities)
			{
				if(saveableEntity.metaData.guid == guid)
				{
					return saveableEntity;
				}
			}
			Debug.LogWarning($"Saveable entity with guid {guid} not found");
			return null;
		}

		public void SetSaveName(string _newSaveName)
		{
			defSaveName = _newSaveName;
		}

		[ContextMenu("Save")]
		public void Save()
		{
			Debug.Log("Saved");
			Debug.Log(SAVE_PATH);
			SaveGame(SAVE_PATH);
		}
		
		[ContextMenu("Save Unique Data")]
		public void Save<T>(T _data, string _fileName)
		{
			defSaveName = _fileName;
			SaveGame(SAVE_PATH, _data);
			Debug.Log("Saved");
		}

		[ContextMenu("Load")]
		public void Load()
		{
			LoadGame(SAVE_PATH);
		}
		
		public void Load(bool _useJson)
		{
			Debug.Log(SAVE_PATH.ToString());
			if(_useJson && File.Exists(SAVE_PATH))
				LoadGame(SAVE_PATH);
		}
		
		public bool Load<T>(ref T data, string _fileName)
		{
			defSaveName = _fileName;
			LoadGame(SAVE_PATH, ref data);
			
			// if data is a TrainingData.TrainingModel
			if (data is TrainingData.TrainingModel)
			{
				TrainingData.TrainingModel model;
				model = (TrainingData.TrainingModel) Convert.ChangeType(data, typeof(TrainingData.TrainingModel));
				return model.biases != null && model.weights != null;
			}

			return true;
		}

		public void DeleteSave()
		{
			if(File.Exists(SAVE_PATH))
			{
				File.Delete(SAVE_PATH);
			}
		}

		private void SaveGame(string _path)
		{
			List<MetaData> metaDataList = MetaData.SaveObject();
			SaveToFile(_path, metaDataList);
		}
		
		private void SaveGame<T>(string _path, T _trainingData)
		{
			SaveToFile(_path, _trainingData);
		}

		private void LoadGame(string path)
		{
			List<MetaData> metaDataList = LoadFromFile<List<MetaData>>(path);
			MetaData.LoadObject(prefabs, metaDataList);
		}
		
		private void LoadGame<T>(string path, ref T data)
		{
			data = LoadFromFile<T>(path);
		}

		private void SaveToFile<T>(string path, T data)
		{
			string json = JsonConvert.SerializeObject(data, jsonSettings);
			if(encrypt)
			{
				string base64 = Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes(json));
				File.WriteAllText(path, base64);
			}
			else
			{
				File.WriteAllText(path, json);
			}
		}
		
		private T LoadFromFile<T>(string path)
		{
			if(File.Exists(path))
			{
				string json = File.ReadAllText(path);
				if(encrypt)
				{
					json = System.Text.Encoding.UTF8.GetString(Convert.FromBase64String(json));
				}

				return json.Length < 10 ? default : JsonConvert.DeserializeObject<T>(json, jsonSettings);
			}
			
			return default;
			
		}
		
		public bool SaveFileExists()
		{
			return File.Exists(SAVE_PATH);
		}

		
		public static float[] ConvertFromVector3(Vector3 vector3)
		{
			float[] values = { vector3.x, vector3.y, vector3.z };
			return values;
		}

		public static Vector3 ConvertToVector3(float[] values)
		{
			return new Vector3(values[0], values[1], values[2]);
		}


		public static float[,] ConvertFromVector3Array(Vector3[] vector3)
		{
			if(vector3 == null)
			{
				return new float[0, 3];
			}

			float[,] values = new float[vector3.Length, 3];
			for(int i = 0; i < vector3.Length; i++)
			{
				values[i, 0] = vector3[i].x;
				values[i, 1] = vector3[i].y;
				values[i, 2] = vector3[i].z;
			}

			return values;
		}

		public static Vector3[] ConvertToVector3Array(float[,] array)
		{
			if(array.Length == 0)
			{
				return null;
			}

			Vector3[] vector3 = new Vector3[array.GetUpperBound(0) + 1];
			for(int i = 0; i < vector3.Length; i++)
			{
				vector3[i] = new Vector3(array[i, 0], array[i, 1], array[i, 2]);
			}

			return vector3;
		}

		
		//Vector2
		public static float[] ConvertFromVector2(Vector2 vector2)
		{
			float[] values = { vector2.x, vector2.y };
			return values;
		}

		public static Vector2 ConvertToVector2(float[] values)
		{
			return new Vector2(values[0], values[1]);
		}

		public static float[,] ConvertFromVector2Array(Vector2[] vector2)
		{
			if(vector2 == null)
			{
				return new float[0, 2];
			}

			float[,] values = new float[vector2.Length, 2];
			for(int i = 0; i < vector2.Length; i++)
			{
				values[i, 0] = vector2[i].x;
				values[i, 1] = vector2[i].y;
			}

			return values;
		}


		public static Vector2[] ConvertToVector2Array(float[,] array)
		{
			if(array.Length == 0)
			{
				return null;
			}

			Vector2[] vector2 = new Vector2[array.GetUpperBound(0) + 1];
			for(int i = 0; i < vector2.Length; i++)
			{
				vector2[i] = new Vector2(array[i, 0], array[i, 1]);
			}

			return vector2;
		}
		
		//Quaternion
		public static float[] ConvertFromQuaternion(Quaternion quaternion)
		{
			float[] values = { quaternion.x, quaternion.y, quaternion.z, quaternion.w };
			return values;
		}

		public static Quaternion ConvertToQuaternion(float[] values)
		{
			return new Quaternion(values[0], values[1], values[2], values[3]);
		}

		public string GetSavePath()
		{
			return SAVE_PATH;
		}
		
	}
}