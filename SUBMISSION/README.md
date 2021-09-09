# build docker image of the agent server

```
$ git clone https://github.com/seq-img-goal-challenge/habitat-lab sig_challenge
$ cd sig_challenge && git checkout seq-objnav
$ docker build -f SUBMISSION/Dockerfile -t sig_agent_srv .
```

Note: the image size is around 10GB at the moment.
