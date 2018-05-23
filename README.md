# TCC

Start with "handle_data/download_data.sh" to download the necessary files from the AQSDataMart and then "handle_data/concat.sh" to concatenate several files from different years into one single file. Next step on the flow is to execute "main.py" which contains the flow for: cleaning, refining, merging, preparing (shifting time steps ahead for prediction), and extracting features from the original dataset.

Now, with the cleaned datasets "gen_models.py" can be edited and executed to create different LSTM and MLP networks.
