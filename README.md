# I. File list
```
.
|    assess.py - Python script of phase 2 assessment
|    Dockerfile - Docker file to run python script
```

# II. How to Run
1. Download the associated data at <insert url later>. Note, that this already contains both the inputs and outputs 
of the phase 2 assessment.
2. Download and Run Docker Desktop. For more information on Docker visit: https://docs.docker.com/desktop/. To ensure 
that it is installed correctly go to the command prompt/terminal and enter $ docker --version
3. Change to the current working directory using command prompt/terminal $ cd <insert_path>\phase_2_assessment
4. Build the docker image by running $ docker build --tag p2_assessment .
5. Run the image and mount the associated data you downloaded in step 1 by running
$docker run -v <path_to_associated_data>\multiobjective_dam_hazard_io:/app_io p2_assessment 
