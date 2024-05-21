# TIMBA code

Steps to replicate the results of the paper:

1- Create the Docker image and enter inside of the container:

- Give execution permissions to the installation file: ```chmod +x setup.sh```

- Run the setup file, this steps can be very slow: ```./setup.sh```

- See the IDs of the running containers: ```docker ps```

- Enter inside of the created container: ```docker exec -it {created_container_id} bash```

2- Replicate the results obtained in the paper:

The experiments are configured with the Hydra library. To run an experiment, simply execute the following command:

```python ./scripts/run_average_experiment.py --config-name {hydra_file}```

Currently, the following files are implemented:

- aqi36.yaml
- metr-la_point.yaml
- metr-la_block.yaml
- pems-bay_point.yaml
- pems-bay_block.yaml

So, for example, if you want to run the experiment with the Metr-La dataset in the Point Missing scenario, you should execute the following command line:

```python ./scripts/run_average_experiment.py --config-name metr-la_point```

If you want to change the size of the time windows, simply go to the yaml file of the experiment you want to run and change the parameter ```scale_window_factor```.

By default, experiments run with the TIMBA model, which is the one proposed in this paper, but you can test CSDI and PriSTI by executing the commands as follows:

- For CSDI:

```python ./scripts/run_average_experiment.py --config-name {hydra_file} model_name=csdi```

- For PriSTI:

```python ./scripts/run_average_experiment.py --config-name {hydra_file} model_name=pristi```
