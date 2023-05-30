namespace SaveLoad
{
	public interface IJSaveable
	{
		public void SaveObjectState(MetaData metaData);

		public void LoadObjectState(MetaData metaData);
	}
}