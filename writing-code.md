# Designing a PR

## Before you start: what must happen

Identify what _must_ happen for a feature to be enacted. This will be "low level" code that, regardless of how you write the feature, must exist somewhere.

For example, if you have a codebase that supports loading only one training dataset and you want to enable concatenating multiple datasets at train-time, then you must add the low-level code to do that concatenation, and will need to be able to configure to allow multiple datasets (here you might decide if what you need is glob/wildcard support, a list, or both).

Next, identify parts of the codebase that _must_ be changed or interacted with directly to enact the feature. For example, if part of the codebase "owns" low-level information required to enact that feature, and that information isn't already exported publicly, then that part of the codebase must be changed (whether that change is to export the information, or to have the low-level feature live there). Again, don't make any design decisions yet, we're just identifying what _must_ happen and where it _must_ happen.

In our dataset-concat example, the low-level concatenation is going to depend directly on the low-level implementation of your training dataset. Also, somewhere in your configuration you'll need to add a way to specify multiple datasets.

Once you have that information figured out, then you can start a design.

## Designing the code

Now that you know what must happen and where it must happen, you want to try to come up with a design.

You want to follow, or at least strive towards, the Single Responsibility Principle. The principle says that each module or class should have responsibility over a single part of the functionality, and that responsibility should be entirely encapsulated by the module or class.

What this means in practice for you is:
 - For each thing that must happen, identify the one part of the codebase responsible for it. If you find part of the responsibility needs to happen in one place and another part somewhere else, then either you need to reframe that responsibility so it makes sense (i.e. if it _really_ is two responsibilities, this is OK), or you need to refactor the code so that the responsibility is in one place.
 - Think carefully if you're updating a class to do an additional thing. If it isn't an option for something that's already the responsibility of the class, you should strongly consider composing a new class against that class, instead of adding new functionality to it.

Somewhat related to the Single Responsibility Principle is the idea that only one part of the code should own the source of truth for a given choice. That means avoiding two configuration options that need to be updated in synchrony - set it in one place, and if a low-level operation needs to know about it, then either the owner of that information needs a method to do that operation, or the owner needs to export that information. You should think carefully about which one of these is appropriate!

Let's again use the dataset-concat example. The two things that needed to happen were the concatenation itself, and specifying multiple datasets in the config.

If you end up deciding that all you really need is glob support, it's possible you won't need to do anything - maybe xarray would natively accept a glob in a way your code "just works". Following these steps would lead you to that conclusion.

If you actually do need concatenation of your datasets themselves, then you might decide:
- Adding a `.concat` method to your `Dataset` class is appropriate since that class owns all the low-level information that operation needs to interact with
- Since your `DatasetConfig` class already handles a lot, it may make sense to write a `ConcatDatasetConfig` class that specifies a list of `DatasetConfig`, so that this new class owns only the concatenation process. This follows the SRP better than adding concatenation into the existing class.

## Delving into complex issues

I might be writing like this is straightforward, but it isn't always. Going into it, you might not be fully familiar with the parts of the codebase that will need to be edited. Still, use these guidelines to figure out which parts of the code you should invest time into understanding.

If you find the design you're coming up with can't satisfy the SRP, or you're finding that you need to pass low-level information between objects at substantially different levels of the codebase, get your teammates involved. You might bring the issue up in standup, or collaborate with someone else on the design.

## Advanced: Changing existing responsibilities

Keeping a responsibility you're adding to one spot in the code is a different challenge than minimizing the size of your PR. The way objects are set up in our code is meant to handle the complexity our code currently has. When you add more complexity, you might need to refactor the existing responsibilities so they better accomodate that new complexity. When you do this, just make sure that each object you're changing still makes sense _on its own_ in the context of _its own responsbility_. Avoid adding low-level complexity to an existing class that's actually associated with your new responsibility. If you're doing that, you should strongly consider composing a new class (in charge of your new responsibility) against that class, instead of adding new functionality to it. And to do that, you may end up splitting up that old class so the code inside it you're "replacing" lives in this new composable concept.
