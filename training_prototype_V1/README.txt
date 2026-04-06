To execute the code properly:
	Need 2 files(included in the folder):
		- week3_air_quality_hourly_20260305_145306.csv 
		- week3_weather_hourly_20260305_145306.csv
	To run:
		`python aqi_model.py'
		- just need to run the script as you would a normal python file

Contributions:
	Cameron
		- Autoencoder
			- found best latent dimension
			- built from scratch
		- Training the model (tuning hyperparameters)
		- Preprocessing
			- about 99% of it
			- Our preprocessing team sent in a bunch of code that does not even work, so we had to do it
			- If you can see the preprocessing teams pipeline, the target feature ('us_aqi') is not even separated from the training features, so with no target, we cannot use a supervised training model
			- Messaged team multiple times, no response to problems, and/or modifications to code until days later, if at all
		- Modified, and finalized pipeline that teammate had originally conceived
	Long
		- (To be filled in by him)
