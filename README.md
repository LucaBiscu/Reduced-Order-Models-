Project: Nonlinear Elliptic problem POD vs PINNs
================================================

Building the container
----------------------

To build the container run either of the following commands:

```bash
docker build -f Dockerfile-cpu --target rbm-base . -t rbm-project:cpu
docker build -f Dockerfile-gpu --target rbm-base . -t rbm-project:gpu 
```

to start the container use:

```bash
podman run -it --name proj -v "Path to code directory":/root:z rbm-project
```

to run the program type:

```bash
python main.py
```



