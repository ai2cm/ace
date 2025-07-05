.. _Builder Pattern:

=====================================
The Builder Pattern: Modern Namelists
=====================================

Introduction
============

The Builder Pattern is a design pattern that lets user configuration change the way a code runs by changing the way objects are constructed.
Those objects are then used by the code, without that code ever knowing about or handling user configuration.

At its core, the builder pattern contains two components:
- A class that represents the configuration, called the Config, which has a build method returning a built instance.
- A class built by the configuration, called the Product or the "instance".

While technically speaking these are defined as two separate classes, the Config is really just the set of variables used by the Product which a user can configure from a configuration file.
For this reason, it is acceptable to tightly couple the configuration and the product, for example by having a method on the configuration that builds the product and having the product take the configuration as an argument. These two classes should be defined in the same file.

This pattern makes it much easier to understand and to change both what the code does and how it is configured, because these two very different types of logic are completely separated and isolated from each other. It also makes it much easier to compartmentalize the configuration. When a class only has access to five attributes and classes always use all their attributes, it's much easier to understand how that class works than if it has access to 100 attributes that are all used in different parts of the code.

Example
=======

The pattern itself can be used with any type of configuration file and any code which loads that configuration into a class. In our code, we use YAML files for configuration, and use the ``dacite`` package to load those files into classes in a type-safe way. Take this example of a yaml configuration file defining an optimizer:

.. testcode::

    with open("config.yaml", "w") as f:
        f.write("""
        optimizer:
          optimizer_type: Adam
          lr: 0.01
          kwargs:
            betas: [0.9, 0.999]
        """)

Let's define a training configuration ``TrainConfig`` containing an ``OptimizerConfig`` to reflect the structure of the configuration file:


.. testcode::

    import dataclasses
    from typing import Literal, Mapping, Any, Iterable

    from torch.optim import Optimizer, Adam, SGD
    from torch.nn import Parameter, Linear

    @dataclasses.dataclass
    class OptimizerConfig:
        """
        Configuration for an optimizer.

        Parameters:
            optimizer_type: The type of optimizer to use.
            lr: The learning rate.
            kwargs: Additional keyword arguments to pass to the optimizer.
        """

        optimizer_type: Literal["Adam", "SGD"] = "Adam"
        lr: float = 0.001
        kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

        def build(self, parameters: Iterable[Parameter]) -> Optimizer:
            if self.optimizer_type == "Adam":
                optimizer = Adam(parameters, lr=self.lr, **self.kwargs)
            elif self.optimizer_type == "SGD":
                optimizer = SGD(parameters, lr=self.lr, **self.kwargs)
            else:
                raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
            return optimizer

    @dataclasses.dataclass
    class TrainingConfig:
        optimizer: OptimizerConfig

We can use ``dacite`` to automatically load the configuration file into this dataclass structure:

.. testcode::

    import dacite
    import yaml

    with open("config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)

    config = dacite.from_dict(TrainingConfig, config_dict, config=dacite.Config(strict=True))
    print(config)

    module = Linear(10, 10)

    optimizer = config.optimizer.build(parameters=module.parameters())
    print(optimizer)

The result is a user-configurable instance we can use to optimize model weights.

.. testoutput::

    TrainingConfig(optimizer=OptimizerConfig(optimizer_type='Adam', lr=0.01, kwargs={'betas': [0.9, 0.999]}))
    Adam (
    Parameter Group 0
        amsgrad: False
        betas: [0.9, 0.999]
        capturable: False
        decoupled_weight_decay: False
        differentiable: False
        eps: 1e-08
        foreach: None
        fused: None
        lr: 0.01
        maximize: False
        weight_decay: 0
    )

.. testcleanup::

    import os
    if os.path.exists("config.yaml"):
        os.remove("config.yaml")

The OptimizerConfig is a Config class with a build method.
The product of that build method is an Optimizer instance from pytorch.
Once we have an Optimizer, the code no longer needs to worry about how that object was configured.

Let's break down what's happening.
The user defines a yaml file representing a TrainConfig.
That yaml contains an "optimizer" section, corresponding to our OptimizerConfig.
Dacite will automatically load nested dataclasses, meaning it automatically loads the yaml data into the nested "optimizer" attribute.
It also means we can add as many other configuration classes under TrainConfig as we want (for example, to define the neural network or how many epochs to train), or even nested configuration classes under those classes (such as a SchedulerConfig class under the optimizer).

Dacite is a very powerful tool for making sure that the configuration is correct and that the code is type-safe.
It reads the configuration file, validates it against the class, and then constructs the class for us.
Normally, adding validation to a configuration file would be extra work as well as code you need to keep up to date if the configuration ever changes, but dacite does it all for us.

As shown in this example, the build method can take in arguments that are only available at runtime, meaning you can build objects that require more than just the provided user configuration.
In this case, the optimizer also takes in the model weights being optimized.
You can also pass in data implied by other configuration classes.
For example, you might have a data configuration specifying data to load, automatically detect the size of the training data (e.g. an image resolution) from that loaded data, and pass that size to another configuration class that builds the neural network.
