# full-model
Create an ML-only climate model

## Development

This code is easiest to develop with access to an NVIDIA GPU. Therefore it is
recommended to develop within an [interactive Beaker session](https://beaker-docs.apps.allenai.org/start/interactive.html).
Once you have logged into a Beaker node and cloned this repo, call
```
make build_beaker_image
```
and then
```
make launch_beaker_session
```
Building the beaker image will be slow the first time, since the base
image must be pulled from NIVDIA's container registry. It will be faster in
subsequent calls on the same day. The code in this repository will be bind-mounted
in the interactive session so that code changes outside the session will be
reflected inside the session.

Alternatively, if you'd like to skip the image building step since you already
have a particular beaker image to use, you can do something like:

```
make VERSION=5df2501e58d5a585b3551979cc8ca1bb9f1585fc launch_beaker_session
```