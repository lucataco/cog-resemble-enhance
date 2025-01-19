# ResembleAI/resemble-enhance Cog model

This is an implementation of [ResembleAI/resemble-enhance](https://github.com/resemble-ai/resemble-enhance) as a [Cog](https://github.com/replicate/cog) model.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own model to [Replicate](https://replicate.com).

## Setup

Download the weight into a resemble folder

    huggingface-cli download ResembleAI/resemble-enhance --local-dir resemble
    
## Basic Usage

Download weights/repo

    git clone https://github.com/resemble-ai/resemble-enhance.git resemble

Run prediction:

    cog predict -i input_audio=@demo.mp3

