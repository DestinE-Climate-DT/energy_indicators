0. activate energy onshore venv


1. I created a directory structure for the data, for he 2 months test by using the function create_stac_catalg.py from energyonshore utils.

data/YYYY/MM


2. I added the ID to every month directory: (same as above)

`python3 /home/froura/repositories/energy_onshore/utils/generate_stac_metadata.py . . `

data/YYYY/MM/EO.XXX.XXX/

After creating this, 


3. I manually added collection_config.json and collection.json example from here collection.json_example
ame for collection_cnfig.json 

4. I run the generate_item_metadata.py you provided, with the message:


(venv_energy_onshore) (base) froura@bsces109861:~/data/a27y/raw/test$ python3 /home/froura/repositories/DestinE-DataLake-Lab/HDA/Usergenerated/generate_item_metadata_bkp.py
Successfully validated stac collection: EO.BSC.DAT.ENERGY_INDICATORS

