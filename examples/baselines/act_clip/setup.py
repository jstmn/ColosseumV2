from setuptools import setup, find_packages

setup(
    name="act_clip",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torchvision",
        "diffusers",
        "tensorboard",
        "wandb",
        "mani_skill",
        "ftfy",
    ],
    description="A minimal setup for ACT_CLIP for ManiSkill",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
