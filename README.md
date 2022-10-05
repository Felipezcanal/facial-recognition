# Facial recognition

## Desenvolvimento

Como iniciar o desenvolvimento

```bash
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements/dev.txt
python3 loadModel.py
```

To run the testing facial recognition with docker:
```bash
docker build -t facial-recognition .
xhost +local:docker
docker run --rm -ti -e DISPLAY=$DISPLAY --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --device="/dev/video1:/dev/video1" facial-recognition 
```
** You may need to change the /dev/video1 to /dev/video0 or whatever may be the index of your webcam