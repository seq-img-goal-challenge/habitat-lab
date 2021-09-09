# build docker image of the habitat-based evaluation client

```
$ git clone https://github.com/seq-img-goal-challenge/habitat-lab sig_challenge
$ cd sig_challenge && git checkout seq-objnav
$ docker build -f docker/Dockerfile -t sig_eval_clt .
```

Note: the image size is around 18GB at the moment.
